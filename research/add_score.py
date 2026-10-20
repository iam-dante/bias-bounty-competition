"""Append one scored file to model.npz.

    python research/add_score.py <history_name.csv> <submission file> <public RMSE>

The vector is aligned to model.npz's GEOID order; a name already present is refused.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(name, path, score):
    with np.load(ROOT / "model.npz", allow_pickle=False) as z:
        art = {k: z[k] for k in z.files}
    assert name not in art["names"], f"{name} is already in model.npz"
    frame = pd.read_csv(ROOT / path, dtype={"GEOID": str}).set_index("GEOID")
    assert len(frame) == 9794 and frame.index.is_unique
    v = frame.loc[art["geoid"], "coverage_gap_score"].to_numpy(dtype=float)
    assert np.isfinite(v).all()
    art["names"] = np.append(art["names"], name).astype(f"<U{max(42, len(name))}")
    art["scores"] = np.append(art["scores"], float(score))
    art["predictions"] = np.column_stack([art["predictions"], v])
    np.savez_compressed(ROOT / "model.npz", **art)
    print(f"model.npz: added {name} ({path}, {float(score):.9f}); {len(art['names'])} scored files")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
