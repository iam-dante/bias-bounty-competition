"""Load confirmed feedback without mutating the frozen historical model artifact."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import solution as S


def load():
    lb=S.Leaderboard()
    for row in json.loads((ROOT/'research/score_feedback.json').read_text()):
        path=ROOT/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        frame=pd.read_csv(path,dtype={'GEOID':str}).set_index('GEOID')
        values=frame.loc[lb.geoid,'coverage_gap_score'].to_numpy()
        assert len(values)==9794 and frame.index.is_unique
        lb.vectors[row['name']]=values
        lb.s[row['name']]=row['rmse']
    lb.public_subset()
    return lb
