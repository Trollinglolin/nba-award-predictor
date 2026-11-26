"""Run the whole pipeline: ingest (cache-first) -> features (+ nba_api cache if present) -> train v1 -> explain -> report
-> v2 feature-group experiments -> v3 pre-registered sets + opt-in candidate -> predict.

The nba_api data is fetched separately (nba_fetch.py); this script never calls stats.nba.com.

    .venv/bin/python -u run_pipeline.py                 # uses data/raw/kaggle_bref/ if present (no network)
    .venv/bin/python -u run_pipeline.py --refresh-data  # re-download the Kaggle export first
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import config as C
import experiments
import experiments_v3
import explain
import features
import ingest
import predict
import report
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ingest.main(["--force"] if args.refresh_data else [])
    raw = features.load_raw()
    table, info = features.build_table(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.TABLE_PATH, index=False)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "build_info.json").write_text(json.dumps(info, indent=2, default=str))
    L = features.label_tables(raw)
    counts = {"all_nba": L["all_nba"].groupby("season").size().to_dict(),
              "all_star": L["all_star"].groupby("season").size().to_dict()}
    train.main(table=table, label_counts=counts)
    explain.main(table=table)
    report.write(info=info)
    v2 = experiments.main(table, counts)
    experiments.write_report(v2, info)
    # v3 (reports/improvement_plan_2024_25.md): two passes share one Bonferroni family (m = 16); the best-family pass
    # runs last and writes models/all_nba_v3.joblib (v1 unless a set is adopted). Archetypes come from
    # ../nba-archetype-pipeline (run its apply_frozen.py first); without them the archetype features are NaN.
    (C.REPORTS_DIR / "v3_logreg").mkdir(exist_ok=True)
    experiments_v3.main(table, counts, family="logreg", n_tests=16, out_reports=C.REPORTS_DIR / "v3_logreg")
    experiments_v3.main(table, counts, n_tests=16)
    experiments_v3.build_candidate(table, counts)
    predict.main([])
    print(f"[pipeline] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
