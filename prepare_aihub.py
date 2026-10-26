"""
Prepare AI Hub strawberry crops (Phytophthora blight vs wilt) for train_optimizers.py.

    python prepare_aihub.py --count-only     # labels only: counts and the planned split (a few minutes)
    python prepare_aihub.py                  # same, then cuts the crops out of the image zips

What it does
  1. Reads every JSON label straight from the TL_/VL_ zips (nothing is extracted to disk).
     The table is cached in aihub_cls/labels_all.csv, so later runs skip this step.
  2. Keeps boxes with disease_status == "Y" (visible symptoms), at least --min-box px wide and high,
     photographed inside the date window that BOTH diseases share (removes the season clue).
  3. Thins the hourly time-lapse: per plant, an image is kept only if it was taken at least
     --gap-hours after the previously kept image (removes near-duplicate frames).
  4. Splits by plant GROUP (plants that share a bed/camera, e.g. 07_021, 07_022, 07_023 -> group 07),
     separately for each disease, aiming at 70 / 15 / 15 of the crops. AI Hub's own split mixes
     photos of the same plant into train and validation, so it is NOT used.
  5. Crops each box (plus a --pad margin) from the TS_/VS_ image zips into
     aihub_cls/<train|validation|test>/<blight|wilt>/.

Outputs (in aihub_cls/): the crop folders, manifest.csv (one row per crop), prep_summary.txt.
All settings are printed in prep_summary.txt; fix them BEFORE training and report them in your methods.
"""
import argparse, io, json, random, shutil, zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "305.시설작물개체이미지 및 시설작물질병이미지" / "01-1.정식개방데이터"
OUT = HERE / "aihub_cls"
CLASS = {"역병": "blight", "시들음병": "wilt"}            # Korean disease name -> folder name
SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count-only", action="store_true", help="stop after the split; do not crop images")
    p.add_argument("--gap-hours", type=float, default=24, help="minimum hours between kept images of one plant")
    p.add_argument("--min-box", type=int, default=32, help="minimum box width and height in pixels")
    p.add_argument("--pad", type=float, default=0.10, help="extra margin around each box (fraction of its size)")
    p.add_argument("--seed", type=int, default=42, help="seed for the group split (fixed for every run)")
    p.add_argument("--reparse", action="store_true", help="re-read the label zips even if labels_all.csv exists")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def fix_name(info):
    """Korean names inside zips are often stored in CP949 rather than UTF-8."""
    if info.flag_bits & 0x800:
        return info.filename
    return info.filename.encode("cp437").decode("cp949", errors="replace")


def load_json(raw):
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp949"))


def find_zips(prefixes):
    zips = sorted(p for p in DATA.rglob("*.zip")
                  if p.name.startswith(prefixes) and any(k in p.name for k in CLASS))
    if not zips:
        raise SystemExit(f"No {prefixes} zips for {list(CLASS)} found under\n{DATA}")
    return zips


# ── 1. read all labels ───────────────────────────────────────────────────────
def read_labels():
    rows, bad = [], []
    for zp in find_zips(("TL_", "VL_")):
        with zipfile.ZipFile(zp) as z:
            members = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".json")]
            print(f"reading {len(members):,} labels from {zp.name}")
            for n, info in enumerate(members, 1):
                lab = load_json(z.read(info))
                img = lab["images"]
                parts = Path(img["fname"]).stem.split("_")      # 딸기_설향_역병_01_001_220720184643
                if len(parts) != 6:
                    bad.append(img["fname"]); continue
                disease = parts[2]
                for k, a in enumerate(lab.get("annotations") or []):
                    x, y, w, h = (float(v) for v in a["bbox"])  # COCO format: top-left x, y, width, height
                    rows.append(dict(
                        label_zip=zp.name[:2], disease=disease, group=parts[3], plant=f"{parts[3]}_{parts[4]}",
                        fname=img["fname"], time=img["date_captured"],
                        img_w=int(float(img["width"])), img_h=int(float(img["height"])),
                        box=k, x=x, y=y, w=w, h=h, status=a.get("disease_status"),
                        category=a.get("category_id")))
                if n % 10000 == 0:
                    print(f"   {n:,} / {len(members):,}")
    if bad:
        print(f"skipped {len(bad)} labels with unexpected file names, e.g. {bad[:3]}")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df.time)
    return df


# ── 2-3. filter and thin ─────────────────────────────────────────────────────
def select(df, args, log):
    df = df[df.disease.isin(CLASS)].copy()
    dup = df.duplicated(["fname", "box"])
    if dup.any():
        log.append(f"Removed {dup.sum():,} duplicate boxes (same image listed in more than one label zip)")
        df = df[~dup]
    log.append("All boxes by disease_status:\n" + df.groupby(["disease", "status"]).size().unstack(fill_value=0).to_string())
    log.append("All boxes by category_id:\n" + df.groupby(["disease", "category"]).size().unstack(fill_value=0).to_string())

    start = df.groupby("disease").time.min().max()
    end = df.groupby("disease").time.max().min()
    log.append(f"Shared date window: {start} to {end}")

    steps = [("symptomatic (status Y)", df.status == "Y"),
             (f"box >= {args.min_box} px", (df.w >= args.min_box) & (df.h >= args.min_box)),
             ("inside shared date window", df.time.between(start, end))]
    keep = pd.Series(True, index=df.index)
    for name, mask in steps:
        keep &= mask
        log.append(f"after {name:28s}: {keep.sum():>7,} boxes  ({df[keep].fname.nunique():,} images)")
    sel = df[keep].copy()

    # time-lapse thinning, one plant at a time
    images = sel.drop_duplicates("fname").sort_values(["disease", "plant", "time"])
    kept, gap = set(), pd.Timedelta(hours=args.gap_hours)
    for _, g in images.groupby(["disease", "plant"]):
        last = None
        for fname, t in zip(g.fname, g.time):
            if last is None or t - last >= gap:
                kept.add(fname); last = t
    sel = sel[sel.fname.isin(kept)].copy()
    log.append(f"after thinning (>= {args.gap_hours:g} h apart) : {len(sel):>7,} boxes  ({sel.fname.nunique():,} images)")
    return sel


# ── 4. group split ───────────────────────────────────────────────────────────
def split_groups(sel, seed):
    rng = random.Random(seed)
    assign = {}
    for disease, d in sel.groupby("disease"):
        sizes = d.groupby("group").size()
        groups = sorted(sizes.index)
        rng.shuffle(groups)
        total, have = sizes.sum(), {s: 0 for s in SPLITS}
        for g in groups:
            # give each group to the split that is furthest below its target share
            s = max(SPLITS, key=lambda s: SPLITS[s] * total - have[s])
            assign[(disease, g)] = s
            have[s] += sizes[g]
    sel["split"] = [assign[(d, g)] for d, g in zip(sel.disease, sel.group)]
    missing = [(s, d) for s in SPLITS for d in CLASS if not ((sel.split == s) & (sel.disease == d)).any()]
    if missing:
        raise SystemExit(f"These split/class combinations are empty: {missing}. Try another --seed.")
    leaked = sel.groupby(["disease", "group"]).split.nunique()
    assert (leaked == 1).all(), "a group ended up in more than one split"
    return sel


# ── 5. crop ──────────────────────────────────────────────────────────────────
def open_oriented(raw, w, h):
    """Return the image in the orientation the label coordinates refer to, or None."""
    im = Image.open(io.BytesIO(raw))
    if im.size == (w, h):
        return im.convert("RGB")
    turned = ImageOps.exif_transpose(im)
    if turned.size == (w, h):
        return turned.convert("RGB")
    return None


def crop_all(sel, pad):
    index = {}
    for zp in find_zips(("TS_", "VS_")):
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if not info.is_dir():
                    index[Path(fix_name(info)).name] = (zp, info)
        print(f"indexed {zp.name}")

    for s in SPLITS:
        if (OUT / s).exists():
            shutil.rmtree(OUT / s)                      # remove crops from any earlier run
    sel = sel.copy()
    sel["crop_path"] = ""
    missing, misoriented, done = [], [], 0

    sel["zip"] = [index[f][0].name if f in index else "" for f in sel.fname]
    missing = sorted(set(sel.loc[sel.zip == "", "fname"]))
    for zip_name, zsel in sel[sel.zip != ""].groupby("zip"):
        with zipfile.ZipFile(index[zsel.fname.iloc[0]][0]) as z:
            for fname, boxes in zsel.groupby("fname"):
                first = boxes.iloc[0]
                im = open_oriented(z.read(index[fname][1]), first.img_w, first.img_h)
                if im is None:
                    misoriented.append(fname); continue
                for i, b in boxes.iterrows():
                    px, py = pad * b.w, pad * b.h
                    box = (max(0, b.x - px), max(0, b.y - py), min(b.img_w, b.x + b.w + px), min(b.img_h, b.y + b.h + py))
                    dest = OUT / b.split / CLASS[b.disease] / f"{b.plant}_{b.time:%y%m%d%H%M%S}_{b.box}.jpg"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    im.crop(tuple(round(v) for v in box)).save(dest, quality=95)
                    sel.at[i, "crop_path"] = str(dest.relative_to(OUT))
                done += 1
                if done % 500 == 0:
                    print(f"   cropped {done:,} images")
    print(f"cropped {done:,} images; {len(missing)} not found in the image zips; {len(misoriented)} size mismatches")
    return sel[sel.crop_path != ""], missing, misoriented


# ── report ───────────────────────────────────────────────────────────────────
def table(sel):
    t = sel.groupby(["split", "disease"]).agg(crops=("fname", "size"), images=("fname", "nunique"),
                                             plants=("plant", "nunique"), groups=("group", "nunique"))
    return t.reindex(SPLITS, level=0).to_string()


def main():
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    cache = OUT / "labels_all.csv"
    if cache.exists() and not args.reparse:
        df = pd.read_csv(cache, parse_dates=["time"], dtype={"group": str, "plant": str})
        print(f"loaded {len(df):,} boxes from {cache.name}")
    else:
        df = read_labels()
        df.to_csv(cache, index=False, encoding="utf-8-sig")

    log = [f"Settings: gap_hours={args.gap_hours:g}, min_box={args.min_box}, pad={args.pad:g}, seed={args.seed}"]
    sel = split_groups(select(df, args, log), args.seed)
    log.append("Planned split (crops, images, plants, groups):\n" + table(sel))
    q = sel[["w", "h"]].quantile([0.1, 0.5, 0.9]).round(0)
    log.append("Box size in pixels (10th / 50th / 90th percentile):\n" + q.to_string())
    log.append("Groups per split:\n" + sel.groupby(["disease", "split"]).group.unique().apply(sorted).to_string())

    if not args.count_only:
        sel, missing, misoriented = crop_all(sel, args.pad)
        log.append("Final crops on disk:\n" + table(sel))
        log.append(f"Images missing from the zips: {len(missing)} {missing[:5]}")
        log.append(f"Images whose size did not match the label: {len(misoriented)} {misoriented[:5]}")
        sel.drop(columns=["zip"]).to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")

    text = "\n\n".join(log)
    (OUT / ("prep_summary_count_only.txt" if args.count_only else "prep_summary.txt")).write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
