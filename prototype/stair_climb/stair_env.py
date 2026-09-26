"""G1 stair-climbing gymnasium env: RL residual torque over a PD hold.

Obs (77,): full qpos (36) + qvel (35) + pelvis height (1)
  + torso up-vector (3) + distance-to-first-step (1) + stair-top-relative z (1).
Action (29,): residual torque in [-1, 1] per actuator, scaled to +/-20% of
  each actuator's ctrl-range half-span, added to the PD hold (gains shared
  with smoke.py: legs kp 400/kd 20, arms kp 40/kd 2).
Reward: +forward/up pelvis progress, +alive/step, -torque^2, -tilt,
  -foot-slip proxy (ankle horizontal speed while low = contact proxy).
Terminate: pelvis h < 0.4 (fall); pelvis past stair end + above top step
  (success). Truncate at 1000 steps (50 Hz control, frame_skip 10).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import gymnasium as gym
import mujoco

import build_scene

DT = 0.002
FRAME_SKIP = 10  # 50 Hz control
MAX_STEPS = 1000
KP_LEG, KD_LEG = 400.0, 20.0  # same as smoke.py
KP_ARM, KD_ARM = 40.0, 2.0  # same as smoke.py
RESIDUAL_FRAC = 0.4  # residual authority: +/-40% (0.2 held gait inside PD well)
FALL_H = 0.4
CORRIDOR_Y = 0.6  # |pelvis y| beyond this = walked around the stairs: terminate, no success
# PD hold target: reset pose + slight ankle dorsiflex. Centers the CoM over
# mid-foot: pure-zero hold tips forward slowly (falls ~frame 1200), +0.03
# stands 10000+ frames with no drift (h=0.791). Gains unchanged (smoke.py).

W_TRACK = 2.0  # velocity-tracking weight (replaces linear W_FWD: stand-still exploit)
TARGET_VX = 0.5  # desired forward speed m/s on flat
W_UP = 3.0
ALIVE = 0.3  # was 1.0: standing still paid +1/step, more than walking earned
W_TORQUE = 1e-5
W_TILT = 1.0
W_SLIP = 0.5
ANKLE_LOW_H = 0.12  # below this, ankle counts as in contact (slip proxy gate)
LIFT_BASE = 0.06  # ankle stance height; clearance counted above this
LIFT_CAP = 0.15  # cap per-foot clearance (clear 0.09 steps + foot size, with margin)
W_LIFT = 8.0  # foot-clearance weight (max ~1.0, half of tracking)
W_ALT = 4.0  # alternation: feet at different heights = stepping (max ~0.6)
ALT_CAP = 0.15
W_YCENTER = 1.0  # lane-centering: -|y|/corridor per step (dense anti-drift)
RESET_NOISE = 0.01  # uniform init noise on actuated joints (0.02 topples 3/5 seeds)


class StairEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rung=None, build=True):
        super().__init__()
        # One rung per process: scene xml is regenerated for the rung.
        # Pass build=False when the parent already built it (parallel envs:
        # workers must NOT rewrite the same xml file concurrently).
        if build:
            build_scene.main([] if rung is None else ["--rung", str(rung)])
        self.model = mujoco.MjModel.from_xml_path(str(build_scene.OUT))
        self.model.opt.timestep = DT
        assert self.model.nu == 29, f"expected 29 actuators, got {self.model.nu}"
        self.data = mujoco.MjData(self.model)
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.ankles = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        # Per-actuator PD gains + residual scales from joint names.
        self.kp = np.zeros(self.model.nu)
        self.kd = np.zeros(self.model.nu)
        for a in range(self.model.nu):
            j = self.model.actuator_trnid[a, 0]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            leg = any(k in name for k in ("hip", "knee", "ankle"))
            self.kp[a], self.kd[a] = (KP_LEG, KD_LEG) if leg else (KP_ARM, KD_ARM)
        half = (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]) / 2
        self.res_scale = RESIDUAL_FRAC * half
        # qpos addresses of each actuated joint for the PD hold target.
        self.qa = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]]
                            for a in range(self.model.nu)])
        self.va = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[a, 0]]
                            for a in range(self.model.nu)])
        # Stair geometry: per-rung from rungs.yaml (module constants are rung2
        # defaults and MUST NOT leak into other rungs' success thresholds).
        if rung is None:
            self.n_steps = build_scene.N_STEPS
            self.rise = build_scene.RISE
            self.run = build_scene.RUN
        else:
            r = build_scene.load_rung(rung)
            self.n_steps, self.rise, self.run = r["n_steps"], r["rise"], r["run"]
        self.start_x = build_scene.START_X
        self.top_z = self.n_steps * self.rise
        self.end_x = self.start_x + self.n_steps * self.run
        nq, nv = self.model.nq, self.model.nv
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(nq + nv + 6,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.q0 = None
        self.steps = 0
        self.prev_pelvis = np.zeros(3)
        self.prev_ankles = np.zeros((2, 3))

    def _up_vector(self):
        return self.data.xmat[self.torso].reshape(3, 3)[:, 2]

    def _obs(self):
        pelvis = self.data.xpos[self.pelvis]
        return np.concatenate([
            self.data.qpos.copy(), self.data.qvel.copy(),
            [pelvis[2]], self._up_vector(),
            [self.start_x - pelvis[0]], [pelvis[2] - self.top_z],
        ])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.q0 is None:
            self.q0 = self.data.qpos.copy()
            for nm in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
                j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                self.q0[self.model.jnt_qposadr[j]] += 0.03
        else:
            self.data.qpos[:] = self.q0
        self.data.qpos[7:] += self.np_random.uniform(-RESET_NOISE, RESET_NOISE, size=self.data.qpos.shape[0] - 7)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.prev_pelvis[:] = self.data.xpos[self.pelvis]
        for i, b in enumerate(self.ankles):
            self.prev_ankles[i] = self.data.xpos[b]
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        residual = action * self.res_scale
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        tau = np.zeros(self.model.nu)
        for _ in range(FRAME_SKIP):
            # PD hold at sim rate (500 Hz, as in smoke.py); residual held per env step.
            tau_pd = self.kp * (self.q0[self.qa] - self.data.qpos[self.qa]) - self.kd * self.data.qvel[self.va]
            tau[:] = np.clip(tau_pd + residual, lo, hi)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        pelvis = self.data.xpos[self.pelvis].copy()
        dt = DT * FRAME_SKIP
        vx = (pelvis[0] - self.prev_pelvis[0]) / dt
        vz = (pelvis[2] - self.prev_pelvis[2]) / dt
        slip = 0.0
        clearance = 0.0
        for i, b in enumerate(self.ankles):
            pos = self.data.xpos[b]
            if pos[2] < ANKLE_LOW_H:
                slip += float(np.linalg.norm(pos[:2] - self.prev_ankles[i][:2]) / dt)
            clearance += min(max(float(pos[2]) - LIFT_BASE, 0.0), LIFT_CAP)
            self.prev_ankles[i] = pos.copy()
        clearance /= len(self.ankles)
        # Lift pays only while moving forward (ramps with speed): no bonus for
        # marching in place, no incentive to hop without progress.
        lift_gate = min(max(vx, 0.0) / TARGET_VX, 1.0)
        # Alternation: |zl-zr| is ~0 when standing/hopping, large when stepping.
        zl = float(self.data.xpos[self.ankles[0]][2])
        zr = float(self.data.xpos[self.ankles[1]][2])
        alternation = min(abs(zl - zr), ALT_CAP) * lift_gate
        self.prev_pelvis[:] = pelvis
        up_z = float(self._up_vector()[2])
        reward = (
            W_TRACK * float(np.exp(-((vx - TARGET_VX) / 0.5) ** 2))
            + W_LIFT * clearance * lift_gate
            + W_ALT * alternation
            - W_YCENTER * min(abs(pelvis[1]) / CORRIDOR_Y, 1.0)
            + W_UP * max(0.0, vz)
            + ALIVE
            - W_TORQUE * float(np.sum(tau ** 2))
            - W_TILT * (1.0 - up_z)
            - W_SLIP * slip
        )
        fell = bool(pelvis[2] < FALL_H)
        success = bool(pelvis[0] > self.end_x and pelvis[2] > self.top_z + 0.2)
        out = bool(abs(pelvis[1]) > CORRIDOR_Y)
        terminated = fell or success or out
        truncated = self.steps >= MAX_STEPS
        info = {"success": success, "pelvis_h": float(pelvis[2]), "pelvis_x": float(pelvis[0])}
        return self._obs(), float(reward), terminated, truncated, info


def validate_rollout(steps=1000, seed=0):
    """Random-action rollout over `steps` env steps (new episode on fall):
    must stay finite; prints mean reward."""
    env = StairEnv()
    obs, _ = env.reset(seed=seed)
    assert np.all(np.isfinite(obs)), "non-finite initial obs"
    rng = np.random.default_rng(seed)
    total, rmin, rmax, done_ep = 0.0, float("inf"), float("-inf"), 0
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=env.model.nu))
        assert np.all(np.isfinite(obs)) and np.isfinite(r), "non-finite step"
        total += r
        rmin, rmax = min(rmin, r), max(rmax, r)
        if term or trunc:
            done_ep += 1
            obs, _ = env.reset()
    print(f"rollout steps={steps} episodes={done_ep + 1} mean_reward={total / steps:.4f} "
          f"min={rmin:.3f} max={rmax:.3f}")
    print("ROLLOUT OK")


if __name__ == "__main__":
    validate_rollout()
