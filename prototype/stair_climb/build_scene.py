"""Build G1-on-stairs scene. Defaults = full stairs; per-rung configs in rungs.yaml.

Rung knobs: n_steps/rise/run (stair geometry), friction (built-scene value;
trainers randomize per-episode inside the rung's friction range), obstacles
(blocks on the approach lane), turn (landing platform past the top step).
"""
import argparse
import pathlib

# Defaults (rung2 / full task): 5-step industrial stair, rise 0.18, run 0.28.
N_STEPS = 5
RISE = 0.18
RUN = 0.28
WIDTH = 1.2
FRICTION = 1.0  # workshop axis: trainers sample friction_lo..friction_hi/episode
START_X = 0.0   # first step front edge; robot spawns at SPAWN_X
SPAWN_X = -1.2

G1 = pathlib.Path(__file__).resolve().parent.parent / \
    "assets/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
OUT = G1.parent / "g1_stairs.xml"
RUNGS_YAML = pathlib.Path(__file__).resolve().parent / "rungs.yaml"


def stairs_geoms(n_steps=N_STEPS, rise=RISE, run=RUN, width=WIDTH,
                 friction=FRICTION, start_x=START_X, obstacles=(), turn=False):
    g = ['<geom name="floor" type="plane" size="8 8 0.1" friction="%.2f %.2f %.2f"/>' % (friction, 0.1, 0.1)]
    for i in range(n_steps):
        top = (i + 1) * rise
        g.append(
            '<geom name="step%d" type="box" size="%.3f %.3f %.3f" pos="%.3f 0 %.3f" '
            'friction="%.2f 0.1 0.1"/>' % (
                i, run / 2, width / 2, top / 2,
                start_x + i * run + run / 2, top / 2, friction))
    for j, ob in enumerate(obstacles):
        h = float(ob.get("h", 0.06))
        g.append(
            '<geom name="obstacle%d" type="box" size="0.150 0.300 %.3f" pos="%.3f 0 %.3f" '
            'friction="%.2f 0.1 0.1"/>' % (j, h / 2, float(ob["x"]), h / 2, friction))
    if turn and n_steps > 0:
        top = n_steps * rise
        end_x = start_x + n_steps * run
        g.append(
            '<geom name="landing" type="box" size="0.500 %.3f %.3f" pos="%.3f 0 %.3f" '
            'friction="%.2f 0.1 0.1"/>' % (
                width / 2, top / 2, end_x + 0.5, top / 2, friction))
    return "\n    ".join(g)


def build_xml(n_steps=N_STEPS, rise=RISE, run=RUN, friction=FRICTION,
              obstacles=(), turn=False):
    src = G1.read_text()
    assert '<body name="pelvis"' in src, "unexpected G1 xml layout"
    # G1 already has a free root joint (floating_base_joint); just spawn pose.
    src = src.replace('<body name="pelvis" pos="0 0 0.793">',
                      f'<body name="pelvis" pos="{SPAWN_X} 0 0.793">', 1)
    assert "</worldbody>" in src
    geoms = stairs_geoms(n_steps=n_steps, rise=rise, run=run,
                         friction=friction, obstacles=obstacles or (), turn=turn)
    return src.replace("</worldbody>", "    " + geoms + "\n  </worldbody>", 1)


def load_rung(n):
    import yaml
    cfg = yaml.safe_load(RUNGS_YAML.read_text())
    key = f"rung{n}"
    assert key in cfg["rungs"], f"{key} missing in {RUNGS_YAML}"
    r = cfg["rungs"][key]
    fr = r.get("friction", [FRICTION, FRICTION])
    return {
        "n_steps": int(r.get("n_steps", N_STEPS)),
        "rise": float(r.get("rise", RISE)),
        "run": float(r.get("run", RUN)),
        "friction": (float(fr[0]) + float(fr[1])) / 2,  # midpoint; trainer samples range/episode
        "obstacles": r.get("obstacles", []) or [],
        "turn": bool(r.get("turn", False)),
        "desc": r.get("desc", ""),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=None, help="curriculum rung 0..3 from rungs.yaml")
    args = ap.parse_args(argv)
    if args.rung is None:
        xml = build_xml()
        tag = f"{N_STEPS} steps, rise={RISE}, run={RUN}, mu={FRICTION}"
    else:
        r = load_rung(args.rung)
        xml = build_xml(n_steps=r["n_steps"], rise=r["rise"], run=r["run"],
                        friction=r["friction"], obstacles=r["obstacles"], turn=r["turn"])
        tag = (f"rung{args.rung}: {r['desc']} "
               f"(n={r['n_steps']}, rise={r['rise']}, mu~{r['friction']:.2f}, "
               f"obstacles={len(r['obstacles'])}, turn={r['turn']})")
    OUT.write_text(xml)
    print(f"wrote {OUT} ({tag})")


if __name__ == "__main__":
    main()
