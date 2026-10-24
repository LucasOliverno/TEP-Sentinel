"""Download the Rieth et al. (2017) TEP dataset and convert it to Parquet.

Source: Harvard Dataverse, doi:10.7910/DVN/6C3JR1 (CC0).

Usage:
    python scripts/download_data.py                 # all four files (~1.4 GB download)
    python scripts/download_data.py --only test     # fault-free + faulty testing
    python scripts/download_data.py --only fault_free
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DATA_FILES  # noqa: E402

DOI = "doi:10.7910/DVN/6C3JR1"
API = "https://dataverse.harvard.edu/api"

# RData file label -> (split, kind)
LABELS = {
    "TEP_FaultFree_Training.RData": ("train", "normal"),
    "TEP_FaultFree_Testing.RData": ("test", "normal"),
    "TEP_Faulty_Training.RData": ("train", "faulty"),
    "TEP_Faulty_Testing.RData": ("test", "faulty"),
}


def _open(url: str, timeout: int):
    # Dataverse rejects urllib's default User-Agent with HTTP 403.
    req = urllib.request.Request(url, headers={"User-Agent": "TEP-Sentinel/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def list_files():
    with _open(f"{API}/datasets/:persistentId/?persistentId={DOI}", 60) as r:
        meta = json.load(r)
    return {f["label"]: f["dataFile"]["id"] for f in meta["data"]["latestVersion"]["files"]}


def download(file_id: int, dest: Path):
    url = f"{API}/access/datafile/{file_id}"
    print(f"  downloading {url} -> {dest}")
    with _open(url, 120) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def convert(rdata: Path, parquet: Path):
    import pyreadr  # imported lazily: only needed here

    print(f"  converting {rdata.name} (this needs a few GB of RAM for the faulty files)...")
    result = pyreadr.read_r(str(rdata))
    df = next(iter(result.values()))
    df["faultNumber"] = df["faultNumber"].astype("int8")
    df["simulationRun"] = df["simulationRun"].astype("int16")
    df["sample"] = df["sample"].astype("int16")
    float_cols = [c for c in df.columns if c.startswith(("xmeas", "xmv"))]
    df[float_cols] = df[float_cols].astype("float32")
    df.sort_values(["faultNumber", "simulationRun", "sample"], inplace=True)
    df.to_parquet(parquet, index=False)
    print(f"  wrote {parquet} ({parquet.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "fault_free", "test"], default="all")
    ap.add_argument("--keep-rdata", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids = list_files()
    for label, key in LABELS.items():
        if args.only == "fault_free" and key[1] != "normal":
            continue
        if args.only == "test" and key[0] != "test":
            continue
        parquet = DATA_DIR / DATA_FILES[key]
        if parquet.exists():
            print(f"[skip] {parquet.name} already exists")
            continue
        print(f"[{label}]")
        rdata = DATA_DIR / label
        if not rdata.exists():
            download(ids[label], rdata)
        convert(rdata, parquet)
        if not args.keep_rdata:
            rdata.unlink()


if __name__ == "__main__":
    main()
