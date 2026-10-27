#!/usr/bin/env python3
"""Download + verify every dataset in manifest.json.  Standard library only (works before any venv exists).

    python datasets/fetch_datasets.py            # download whatever is missing, verify checksums
    python datasets/fetch_datasets.py --verify   # only re-hash local files against CHECKSUMS.sha256
    python datasets/fetch_datasets.py --only crop_recommendation_kaggle_atharvaingle
    python datasets/fetch_datasets.py --docs     # regenerate SOURCE.md files and DATASETS.md from the manifest
"""
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
CHECKS = HERE / "CHECKSUMS.sha256"
UA = {"User-Agent": "annu_ml-fetch/1.0 (student project)"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_get(url: str, tries: int = 3, timeout: int = 120) -> bytes:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def read_checks() -> dict:
    out = {}
    if CHECKS.exists():
        for line in CHECKS.read_text().splitlines():
            if line.strip():
                h, p = line.split("  ", 1)
                out[p] = h
    return out


def write_checks(d: dict):
    CHECKS.write_text("".join(f"{h}  {p}\n" for p, h in sorted(d.items())))


def fetch_one(ds: dict, checks: dict) -> list:
    """Return [(relative_path, sha256)] for files of this dataset, downloading missing ones."""
    folder = HERE / ds["dir"]
    folder.mkdir(parents=True, exist_ok=True)
    done = []

    def store(name: str, data: bytes, want: str | None):
        got = sha256(data)
        if want and got != want:
            raise RuntimeError(f"CHECKSUM MISMATCH for {ds['dir']}/{name}: got {got}, expected {want}")
        (folder / name).write_bytes(data)
        done.append((f"{ds['dir']}/{name}", got))
        print(f"  downloaded {ds['dir']}/{name}  {len(data):,} B  sha256 {got[:12]}.. "
              f"{'(verified)' if want else '(first fetch: recorded)'}")

    def have(name: str, want: str | None):
        p = folder / name
        if not p.exists():
            return False
        got = sha256(p.read_bytes())
        exp = want or checks.get(f"{ds['dir']}/{name}")
        if exp and got != exp:
            print(f"  ! {ds['dir']}/{name} exists but checksum differs -> re-downloading")
            return False
        done.append((f"{ds['dir']}/{name}", got))
        print(f"  ok         {ds['dir']}/{name}  (present, checksum {'verified' if exp else 'recorded'})")
        return True

    kind = ds["kind"]
    if kind == "zip":
        if not have(ds["save_as"], ds["sha256"]):
            try:
                z = zipfile.ZipFile(io.BytesIO(http_get(ds["url"])))
                names = z.namelist()
                assert all(not n.startswith("/") and ".." not in n for n in names), "unsafe zip member"
                data = z.read(ds["member"])
                print(f"  (primary source: {ds['url'].split('/')[2]})")
            except Exception as e:  # noqa: BLE001
                print(f"  ! primary source failed ({e}); using fallback mirror")
                data = http_get(ds["fallback_csv_url"])
            store(ds["save_as"], data, ds["sha256"])
    elif kind == "url":
        if not have(ds["save_as"], ds.get("sha256")):
            store(ds["save_as"], http_get(ds["url"]), ds.get("sha256"))
    elif kind == "mendeley":
        if not all(have(f["save_as"], None) for f in ds["files"]):
            meta = json.loads(http_get(f"https://data.mendeley.com/public-api/datasets/{ds['id']}"))
            for spec in ds["files"]:
                hit = [f for f in meta["files"] if f["filename"].lower().endswith(spec["suffix"])]
                if len(hit) != 1:
                    raise RuntimeError(f"{ds['id']}: expected exactly one *{spec['suffix']} file, got {len(hit)}")
                cd = hit[0]["content_details"]
                store(spec["save_as"], http_get(cd["download_url"]), cd["sha256_hash"])
    else:
        raise RuntimeError(f"unknown kind {kind}")
    return done


def source_md(ds: dict) -> str:
    lines = [f"# {ds['title']}", "", f"- **Role:** {ds['role']}", f"- **Landing page:** {ds['landing_url']}"]
    if ds.get("reference_notebook"):
        lines.append(f"- **Reference notebook:** {ds['reference_notebook']}")
    lines += [f"- **License:** {ds['license']}", f"- **Rows:** {ds['rows']:,}", f"- **Columns:** {ds['columns']}",
              f"- **Classes / target:** {ds['classes']}"]
    if ds["kind"] == "zip":
        lines += [f"- **Download:** `{ds['url']}` (zip, no login needed when this was written); fallback `{ds['fallback_csv_url']}`",
                  f"- **Expected sha256:** `{ds['sha256']}`"]
    elif ds["kind"] == "url":
        lines += [f"- **Download:** `{ds['url']}`", f"- **Expected sha256:** `{ds['sha256']}`"]
    else:
        lines.append(f"- **Download:** Mendeley public API `https://data.mendeley.com/public-api/datasets/{ds['id']}` "
                     "(no login; file hash is checked against the one Mendeley publishes)")
    lines += ["", ds["notes"], "", "_Generated from datasets/manifest.json by `fetch_datasets.py --docs`. Re-fetch: "
              f"`python datasets/fetch_datasets.py --only {ds['dir']}`._", ""]
    return "\n".join(lines)


def write_docs():
    for ds in MANIFEST["datasets"]:
        (HERE / ds["dir"]).mkdir(parents=True, exist_ok=True)
        (HERE / ds["dir"] / "SOURCE.md").write_text(source_md(ds), encoding="utf-8")
    rows = ["# Dataset index", "", "Generated from `datasets/manifest.json`. Cite dataset authors in any report (CC BY needs attribution).", "",
            "| Folder | Role | Rows | License | Source |", "|---|---|---|---|---|"]
    for ds in MANIFEST["datasets"]:
        lic = ds["license"].split(" (")[0].split(".")[0]
        rows.append(f"| `datasets/{ds['dir']}` | {ds['role']} | {ds['rows']:,} | {lic} | [{ds['landing_url'].split('//')[1][:40]}]({ds['landing_url']}) |")
    rows += ["", "## Found but not downloaded", "", "| Dataset | Why |", "|---|---|"]
    rows += [f"| [{n['name']}]({n['url']}) | {n['reason']} |" for n in MANIFEST["not_downloaded"]]
    rows += ["", "Per-dataset details, checksums and caveats: `datasets/<folder>/SOURCE.md`.", ""]
    (HERE / "DATASETS.md").write_text("\n".join(rows), encoding="utf-8")
    print("wrote SOURCE.md files and datasets/DATASETS.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="fetch only this dataset folder")
    ap.add_argument("--verify", action="store_true", help="verify local files only, download nothing")
    ap.add_argument("--docs", action="store_true", help="regenerate SOURCE.md files + DATASETS.md")
    a = ap.parse_args()

    if a.docs:
        write_docs()
        return
    checks = read_checks()
    if a.verify:
        bad = 0
        for rel, want in sorted(checks.items()):
            p = HERE / rel
            got = sha256(p.read_bytes()) if p.exists() else "MISSING"
            ok = got == want
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {rel}")
        sys.exit(1 if bad else 0)

    for ds in MANIFEST["datasets"]:
        if a.only and ds["dir"] != a.only:
            continue
        print(f"[{ds['dir']}]")
        for rel, h in fetch_one(ds, checks):
            checks[rel] = h
    write_checks(checks)
    write_docs()
    print(f"\n{len(checks)} files verified/recorded in datasets/CHECKSUMS.sha256")


if __name__ == "__main__":
    main()
