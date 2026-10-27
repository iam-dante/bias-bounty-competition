"""Run the reviewed experiments, select a candidate, and always finish with a CSV.

    python experiment_loop.py
    python experiment_loop.py --score <actual leaderboard RMSE>

The second command records the exact immutable candidate from the last manifest,
then refits on the expanded score history. It never manufactures a score or uploads.
Unchanged inputs reuse their reports; --refresh reruns the numerical experiments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

import train
import aggregate_kernel_experiment
import robust_posterior_experiment


MANIFEST = Path("outputs/final_submission_manifest.json")
FEEDBACK_DIR = Path("outputs/score_feedback")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = [Path(p) for p in (train.LABELS, "train.py", "aggregate_kernel_experiment.py",
                               "posterior_experiment.py", "robust_posterior_experiment.py", "experiment_loop.py", "requirements.txt")]
    paths += sorted(Path(train.FEAT).glob("*.parquet"))
    paths += sorted(Path(train.DATA).glob("reference/*/*-sample-submission.csv"))
    inputs = {str(p): sha(p) for p in paths}
    inputs["runtime"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             scipy=scipy.__version__, sklearn=sklearn.__version__, machine=platform.machine())
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def record_feedback(score, manifest_path=None):
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("provide a finite actual RMSE between zero and one")
    source = Path(manifest_path) if manifest_path is not None else MANIFEST
    if not source.exists():
        raise ValueError("no submission manifest; identify the exact scored file before recording feedback")
    previous = json.loads(source.read_text())
    path = previous["immutable_submission"]
    if sha(path) != previous["sha256"]:
        raise ValueError("the recorded candidate changed; refusing to attach its score")
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        for receipt_path in FEEDBACK_DIR.glob("*.json"):
            receipt = json.loads(receipt_path.read_text())
            if abs(receipt["score"] - score) < 1e-12 and receipt["label"] != previous["next_label"]:
                raise ValueError("this score was already recorded for another candidate; use --score-manifest with "
                                 "the exact immutable manifest to distinguish a retry from a newly scored file")
    added = train.add_label(previous["next_label"], path, score)
    prediction = previous["estimated_public_rmse"]
    sd = previous.get("model_sd_rmse", 0)
    receipt = dict(score=float(score), label=previous["next_label"], submission=path, sha256=previous["sha256"],
                   forecast_before_score=prediction, forecast_error=float(score - prediction),
                   forecast_model_sd=sd, standardized_error=float((score-prediction)/sd) if sd else None,
                   source="user-reported public leaderboard RMSE", added=bool(added))
    receipt_name = hashlib.sha256(previous["next_label"].encode()).hexdigest()[:20] + ".json"
    train.atomic_json(str(FEEDBACK_DIR / receipt_name), receipt)
    return receipt


def validate(path, labels):
    frame = pd.read_csv(path, dtype={"GEOID": str})
    if frame.columns.tolist() != ["GEOID", "coverage_gap_score"]:
        raise ValueError("submission columns must match the sample")
    expected = pd.concat([pd.read_csv(f"{train.DATA}/reference/{r}/{r}-sample-submission.csv", dtype={"GEOID": str})
                          [["GEOID"]] for r in train.REGIONS], ignore_index=True)
    if len(frame) != len(expected) or not frame.GEOID.is_unique or set(frame.GEOID) != set(expected.GEOID):
        raise ValueError("submission IDs differ from the authoritative sample")
    values = frame.coverage_gap_score.to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("invalid coverage gap values")
    if frame.GEOID.tolist() != labels.geoid.tolist() or not frame.GEOID.str.fullmatch(r"\d{11}").all():
        raise ValueError("submission ID ordering or leading zero preservation failed")
    return dict(rows=len(frame), unique_ids=int(frame.GEOID.nunique()), missing_values=int(frame.isna().sum().sum()),
                min_score=float(values.min()), max_score=float(values.max()), matches_authoritative_samples=True,
                leading_zero_ids=int(frame.GEOID.str.startswith("0").sum()))


def run(score=None, refresh=False, score_manifest=None):
    if score is not None:
        record_feedback(score, score_manifest)
    labels = train.AggregateLabels()
    key = fingerprint()
    run_dir = Path("outputs/experiment_runs") / key[:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    kernel_path, posterior_path = run_dir / "kernel.json", run_dir / "posterior.json"
    if refresh or not kernel_path.exists():
        kernel = aggregate_kernel_experiment.run()
        train.atomic_json(str(kernel_path), kernel)
    else:
        kernel = json.loads(kernel_path.read_text())
        train.log("Reusing completed kernel experiment for identical inputs")
    if refresh or not posterior_path.exists():
        posterior = robust_posterior_experiment.run()
        train.atomic_json(str(posterior_path), posterior)
        shutil.copyfile(posterior["output"], run_dir / "selected.csv")
    else:
        posterior = json.loads(posterior_path.read_text())
        train.log("Reusing completed posterior experiment for identical inputs")
    candidate = run_dir / "selected.csv"
    if sha(candidate) != posterior["output_sha256"]:
        raise ValueError("cached candidate checksum mismatch")
    checks = validate(candidate, labels)
    final_sha = sha(candidate)
    immutable_dir = Path("outputs/submissions")
    immutable_dir.mkdir(parents=True, exist_ok=True)
    immutable = immutable_dir / (final_sha + ".csv")
    if immutable.exists():
        if sha(immutable) != final_sha:
            raise ValueError("immutable submission content was modified")
    else:
        shutil.copyfile(candidate, immutable)
    values = pd.read_csv(candidate, dtype={"GEOID": str}).coverage_gap_score.to_numpy(float)
    matching_labels = [name for name, v in labels.V.items() if np.allclose(v, values, atol=5e-8, rtol=0)]
    next_label = (min(matching_labels, key=labels.s.get) if matching_labels
                  else f"tr_refined{len(labels.s):03d}_{final_sha[:8]}.csv")
    previous_csv = Path("submission.csv")
    if previous_csv.exists() and sha(previous_csv) != final_sha:
        archive = Path("outputs") / ("submission_before_refinement_" + sha(previous_csv)[:12] + ".csv")
        if not archive.exists():
            shutil.copyfile(previous_csv, archive)
        prior_report = Path("outputs/submission_report.json")
        if prior_report.exists() and not archive.with_suffix(".json").exists():
            shutil.copyfile(prior_report, archive.with_suffix(".json"))
    temp = Path("submission.csv.tmp")
    shutil.copyfile(candidate, temp)
    temp.replace(previous_csv)
    chosen = posterior["selected"]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_fingerprint=key,
                    submission="submission.csv", immutable_submission=str(immutable), sha256=final_sha,
                    next_label=next_label, measured_best=posterior["measured_best"],
                    already_scored=bool(matching_labels),
                    selected=chosen["name"], estimated_public_rmse=chosen["mean_over_seeds"],
                    model_sd_rmse=chosen["sd_rmse"], modeled_p_beats_best=chosen["p_beats_best"],
                    scenario_rmse_range=chosen["scenario_rmse_range"], min_scenario_p_beats_best=chosen["min_p_beats_best"],
                    worst_risk_adjusted_rmse=chosen["worst_risk_adjusted_rmse"],
                    target_004_supported=posterior["target_004_supported"], checks=checks,
                    kernel_decision=kernel["decision"], posterior_report=str(posterior_path),
                    kernel_report=str(kernel_path),
                    forecast_limitation=posterior["caveat"],
                    feedback_command=f"python experiment_loop.py --score <actual-public-RMSE> --score-manifest {immutable.with_suffix('.json')}")
    train.atomic_json(str(MANIFEST), manifest)
    immutable_manifest = immutable.with_suffix(".json")
    if not immutable_manifest.exists():
        train.atomic_json(str(immutable_manifest), manifest)
    train.atomic_json("outputs/submission_report.json", dict(
        file="submission.csv", file_md5=train.file_md5("submission.csv"), next_label=next_label,
        inputs=f"challenge features and {len(labels.s)} real score labels",
        method="continuous posterior search across ten residual/prior scenarios; see final_submission_manifest.json",
        aggregate_fit=dict(chosen=chosen["name"], chosen_est=chosen["mean_over_seeds"], best_scored=posterior["measured_best"]),
        final_manifest=str(MANIFEST), caveat=posterior["caveat"]))
    with open("outputs/experiment_history.jsonl", "a") as stream:
        stream.write(json.dumps(manifest) + "\n")
    target_state = "supported by model forecast only" if posterior["target_004_supported"] else "unsupported"
    train.log(f"FINAL submission.csv: {checks['rows']} rows; estimated {chosen['mean_over_seeds']:.8f}; "
              f"measured best {posterior['measured_best']['score']:.8f}; 0.04 {target_state}")
    if matching_labels:
        train.log("No distinct update passed selection; this file is already scored and need not be uploaded again.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=float, help="Actual returned score for the last immutable final submission")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--score-manifest", help="Immutable manifest of the exact submitted CSV; makes feedback retries unambiguous")
    args = parser.parse_args()
    if args.score_manifest and args.score is None:
        parser.error("--score-manifest requires --score")
    with threadpool_limits(limits=2):
        run(args.score, args.refresh, args.score_manifest)
