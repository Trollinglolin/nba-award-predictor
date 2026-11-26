"""Download (cache-first) and validate the Kaggle `sumitrodatta/nba-aba-baa-stats` export.

Usage:
    .venv/bin/python ingest.py            # download only if data/raw/kaggle_bref/ is missing
    .venv/bin/python ingest.py --force    # re-download (records the new date)
    .venv/bin/python ingest.py --zip PATH # use a zip you downloaded in a browser

The raw files are never modified. The data derives from Basketball-Reference; do not redistribute it.
"""
import argparse
import datetime as dt
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

import config as C


def download(dest: Path, retries: int = 5, timeout: int = 120) -> None:
    for attempt in range(1, retries + 1):
        try:
            print(f"[ingest] GET {C.KAGGLE_URL} (attempt {attempt}/{retries})", flush=True)
            with requests.get(C.KAGGLE_URL, stream=True, timeout=timeout, allow_redirects=True) as r:
                print(f"[ingest] HTTP {r.status_code}", flush=True)
                if 400 <= r.status_code < 500:  # not transient: do not retry, keep the existing cache
                    sys.exit(f"[ingest] HTTP {r.status_code} from Kaggle (existing data/raw is untouched). Fallbacks:\n"
                             f'  curl -L -m 120 -o kaggle_bref.zip "{C.KAGGLE_URL}"\n'
                             "  (or a browser download, or: kaggle datasets download -d sumitrodatta/nba-aba-baa-stats)\n"
                             "then: .venv/bin/python ingest.py --zip kaggle_bref.zip --origin 'curl from Kaggle, <date>'\n"
                             "Note: on 2026-11-12 curl got HTTP 200 while python-requests got 404 for the same URL.")
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
            print(f"[ingest] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
            return
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"[ingest] failed: {type(exc).__name__}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    sys.exit("[ingest] download failed after retries")


def unzip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():  # refuse path traversal / nested paths
            if Path(name).name != name or name.startswith("."):
                raise ValueError(f"unexpected member in zip: {name!r}")
        zf.extractall(out_dir)


def validate(raw_dir: Path) -> dict:
    """Schema + season-range checks. Returns {file: max_season} for the label files."""
    problems, max_season = [], {}
    for fname, cols in C.EXPECTED_COLUMNS.items():
        path = raw_dir / fname
        if not path.exists():
            problems.append(f"missing file {fname}")
            continue
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            problems.append(f"{fname}: missing columns {missing}")
        max_season[fname] = int(df["season"].max())
        print(f"[ingest] {fname:36s} rows={len(df):6d} seasons {int(df['season'].min())}-{int(df['season'].max())}",
              flush=True)
    if problems:
        raise SystemExit("[ingest] schema problems:\n  " + "\n  ".join(problems))
    return max_season


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if the data exists")
    ap.add_argument("--zip", type=Path, help="use an already-downloaded zip")
    ap.add_argument("--origin", default="", help="free-text note on where --zip came from (recorded in DOWNLOAD_INFO.txt)")
    args = ap.parse_args(argv)

    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    have = C.KAGGLE_DIR.exists() and (C.KAGGLE_DIR / "Advanced.csv").exists()
    if args.zip:
        shutil.copy(args.zip, C.KAGGLE_ZIP)
        unzip(C.KAGGLE_ZIP, C.KAGGLE_DIR)
        source = f"local zip {args.zip.name}" + (f" ({args.origin})" if args.origin else "")
    elif args.force or not have:
        download(C.KAGGLE_ZIP)
        unzip(C.KAGGLE_ZIP, C.KAGGLE_DIR)
        source = C.KAGGLE_URL
    else:
        print(f"[ingest] {C.KAGGLE_DIR} exists; skipping download (use --force to refresh)", flush=True)
        source = None

    max_season = validate(C.KAGGLE_DIR)
    if source is not None:
        with zipfile.ZipFile(C.KAGGLE_ZIP) as zf:
            file_dates = sorted({dt.date(*zf.getinfo(f).date_time[:3]).isoformat() for f in C.EXPECTED_COLUMNS})
        lines = [f"downloaded: {dt.datetime.now().isoformat(timespec='seconds')}",
                 f"source: {source}",
                 f"file modification dates inside the zip: {', '.join(file_dates)}",
                 "max season per file (season = ENDING year):"]
        lines += [f"  {f}: {s}" for f, s in max_season.items()]
        C.DOWNLOAD_INFO.write_text("\n".join(lines) + "\n")
        print(f"[ingest] wrote {C.DOWNLOAD_INFO}", flush=True)
    labels_max = {f: max_season[f] for f in C.LABEL_FILES}
    print(f"[ingest] label freshness: {labels_max}", flush=True)
    return max_season


if __name__ == "__main__":
    main()
