"""Skeleton smoke test: G1 stands (PD hold) in front of stairs.

Bar for tonight: scene builds, sim steps 1000 frames, no NaN, robot upright.
Usage: smoke.py [--rung N]  (rung scene from rungs.yaml; default full stairs)
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import mujoco

import build_scene

DT = 0.002
STEPS = 1000
KP_LEG, KD_LEG = 400.0, 20.0
KP_ARM, KD_ARM = 40.0, 2.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=None, help="curriculum rung 0..3")
    args = ap.parse_args(argv)
    build_scene.main([] if args.rung is None else ["--rung", str(args.rung)])
    model = mujoco.MjModel.from_xml_path(str(build_scene.OUT))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    q0 = data.qpos.copy()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    model.opt.timestep = DT
    for _ in range(STEPS):
        tau = np.zeros(model.nu)
        # PD per actuator via its joint
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            qa, va = model.jnt_qposadr[j], model.jnt_dofadr[j]
            kp, kd = (KP_LEG, KD_LEG) if any(k in name for k in ("hip", "knee", "ankle")) else (KP_ARM, KD_ARM)
            tau[a] = kp * (q0[qa] - data.qpos[qa]) - kd * data.qvel[va]
        data.ctrl[:] = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
    h = float(data.xpos[pelvis, 2])
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    print(f"rung={args.rung} steps={STEPS} pelvis_h={h:.3f} finite={finite}")
    assert finite, "NaN in state"
    assert h > 0.5, f"robot fell (h={h:.3f})"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
