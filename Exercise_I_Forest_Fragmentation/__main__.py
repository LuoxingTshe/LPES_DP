"""Exercise I – forest fragmentation in the Canton of Bern.

Run from anywhere:
    python Exercise_I_Forest_Fragmentation               # all stages
    python Exercise_I_Forest_Fragmentation --stages 2 3  # selected stages
    python Exercise_I_Forest_Fragmentation --config other.yaml
"""
from pathlib import Path
import argparse
import yaml

from V_scripts import t1_grid, t2_metrics, t3_edge, t4_drivers, t5_maps

HERE = Path(__file__).resolve().parent
STAGES = {1: t1_grid, 2: t2_metrics, 3: t3_edge, 4: t4_drivers, 5: t5_maps}


def load_cfg(path=HERE / "cfg.yaml"):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["paths"] = {k: (path.parent / v).resolve() for k, v in cfg["paths"].items()}
    for k, p in cfg["paths"].items():
        if k != "raw":
            p.mkdir(parents=True, exist_ok=True)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="+", default=list(STAGES))
    ap.add_argument("--config", default=HERE / "cfg.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    for s in args.stages:
        print(f"== Stage {s}: {STAGES[s].__name__}")
        STAGES[s].run(cfg)
