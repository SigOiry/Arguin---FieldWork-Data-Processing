#!/usr/bin/env python
"""Extract Sentinel-2 reflectance pixels for AOI polygons.

Default inputs:
  Data/S2/Sen2Cor
  Data/S2/*.SAFE for the requested 2026 date
  Data/S2/ACOLITE
  Data/shp/TimeSeries_Polygons_Matchup.shp

Default outputs:
  Data/Atm_Compare_AOI/{Sen2Cor,ACOLITE}/YYYY-MM-DD.csv

The default date filter extracts all 2025 scenes plus 2026-03-27.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


METADATA_COLUMNS = [
    "date",
    "AOI",
    "atmospheric_correction",
    "scene_id",
    "source",
    "pixel_id",
    "row",
    "col",
    "x",
    "y",
]

S2_BAND_PATTERN = re.compile(r"_(B(?:0[1-9]|1[0-2]|8A))_", re.IGNORECASE)
ACOLITE_RHOS_PATTERN = re.compile(r"(?i)(rhos(?:_l2a)?_(?P<wavelength>\d+(?:\.\d+)?))")
DATE_PATTERNS = [
    re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})_(\d{2})_(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{8})T\d{6}"),
    re.compile(r"(?<!\d)(\d{8})(?!\d)"),
]

# Approximate Sentinel-2 band centres used only to map ACOLITE rhos wavelengths
# back to comparable Sentinel-2 band columns.
ACOLITE_BAND_CENTRES = {
    "B01": 443.0,
    "B02": 492.0,
    "B03": 560.0,
    "B04": 665.0,
    "B05": 705.0,
    "B06": 740.0,
    "B07": 783.0,
    "B08": 833.0,
    "B8A": 865.0,
    "B09": 945.0,
    "B11": 1610.0,
    "B12": 2190.0,
}

BAND_ORDER = {
    "B01": 1,
    "B02": 2,
    "B03": 3,
    "B04": 4,
    "B05": 5,
    "B06": 6,
    "B07": 7,
    "B08": 8,
    "B8A": 8.5,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}


@dataclass(frozen=True)
class BandSource:
    band: str
    open_path: str
    display_path: str
    scale: float = 1.0
    offset: float = 0.0
    nodata: Optional[float] = None


@dataclass(frozen=True)
class Scene:
    correction: str
    date: str
    scene_id: str
    source: str
    bands: Dict[str, BandSource]


def require_dependencies() -> None:
    missing = []
    for package in ("fiona", "rasterio"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise SystemExit(
            "Missing required geospatial Python packages: "
            + ", ".join(missing)
            + ". Install them in the environment used to run this script, for example: "
            + "python -m pip install rasterio fiona"
        )


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def normalize_date(text: str) -> str:
    date = parse_date_from_text(text)
    if date is None:
        raise ValueError(f"Could not parse date: {text}")
    return date


def parse_date_from_text(text: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if len(match.group(1)) == 8:
            ymd = match.group(1)
            return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def parse_date_from_path(path: Path) -> Optional[str]:
    for part in [path.name] + [parent.name for parent in path.parents]:
        date = parse_date_from_text(part)
        if date is not None:
            return date
    return None


def include_date(date: str, years: Sequence[int], dates: Sequence[str], all_dates: bool) -> bool:
    if all_dates:
        return True
    if date in dates:
        return True
    return int(date[0:4]) in set(years)


def normalize_s2_band(band: str) -> str:
    band = band.upper()
    if band == "B8A":
        return band
    if band.startswith("B") and band[1:].isdigit():
        return f"B{int(band[1:]):02d}"
    return band


def band_sort_key(band: str) -> Tuple[float, str]:
    return BAND_ORDER.get(band, 999), band


def local_name(element: ET.Element) -> str:
    return element.tag.split("}")[-1]


def text_of_first(root: ET.Element, tag: str) -> Optional[str]:
    for element in root.iter():
        if local_name(element) == tag and element.text is not None:
            return element.text.strip()
    return None


def text_of_child(root: ET.Element, tag: str) -> Optional[str]:
    for child in root:
        if local_name(child) == tag and child.text is not None:
            return child.text.strip()
    return None


def parse_sen2cor_metadata_xml(root: ET.Element, fallback_date: Optional[str]) -> Tuple[Optional[str], float, Optional[float], Dict[str, float]]:
    date = text_of_first(root, "PRODUCT_START_TIME") or text_of_first(root, "DATATAKE_SENSING_START")
    date = date[0:10] if date is not None else fallback_date

    quant_text = text_of_first(root, "BOA_QUANTIFICATION_VALUE") or text_of_first(root, "QUANTIFICATION_VALUE")
    quant = float(quant_text) if quant_text is not None else 10000.0

    nodata = None
    for special in root.iter():
        if local_name(special) != "Special_Values":
            continue
        label = text_of_child(special, "SPECIAL_VALUE_TEXT")
        value = text_of_child(special, "SPECIAL_VALUE_INDEX")
        if label == "NODATA" and value is not None:
            nodata = float(value)

    band_id_to_name: Dict[str, str] = {}
    for element in root.iter():
        if local_name(element) != "Spectral_Information":
            continue
        band_id = element.attrib.get("bandId")
        physical = element.attrib.get("physicalBand")
        if band_id is not None and physical is not None:
            band_id_to_name[band_id] = normalize_s2_band(physical)

    offsets: Dict[str, float] = {}
    for element in root.iter():
        if local_name(element) != "BOA_ADD_OFFSET":
            continue
        band_id = element.attrib.get("band_id")
        if band_id is None or element.text is None:
            continue
        band = band_id_to_name.get(band_id)
        if band is not None:
            offsets[band] = float(element.text)

    return date, quant, nodata, offsets


def parse_sen2cor_metadata_file(metadata_file: Path, fallback_date: Optional[str]) -> Tuple[Optional[str], float, Optional[float], Dict[str, float]]:
    root = ET.parse(metadata_file).getroot()
    return parse_sen2cor_metadata_xml(root, fallback_date)


def parse_sen2cor_metadata_zip(zip_path: Path, metadata_member: str, fallback_date: Optional[str]) -> Tuple[Optional[str], float, Optional[float], Dict[str, float]]:
    with zipfile.ZipFile(zip_path) as archive:
        xml_bytes = archive.read(metadata_member)
    root = ET.fromstring(xml_bytes)
    return parse_sen2cor_metadata_xml(root, fallback_date)


def find_first_existing(patterns: Iterable[str], root: Path) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(root.glob(pattern), key=lambda p: str(p).lower())
        if matches:
            return matches[0]
    return None


def find_sen2cor_products(roots: Sequence[Path]) -> List[Path]:
    products: List[Path] = []
    seen = set()

    def add_product(path: Path) -> None:
        key = str(path.resolve()).lower()
        if key not in seen:
            products.append(path)
            seen.add(key)

    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".zip":
            add_product(root)
            continue
        if root.is_dir() and root.name.lower().endswith(".safe"):
            add_product(root)
            continue
        if not root.is_dir():
            continue

        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            for dirname in list(dirs):
                if dirname.lower().endswith(".safe"):
                    add_product(current_path / dirname)
            dirs[:] = [dirname for dirname in dirs if not dirname.lower().endswith(".safe")]

            for filename in files:
                if filename.lower().endswith(".zip"):
                    add_product(current_path / filename)

    return sorted(products, key=lambda p: str(p).lower())


def sen2cor_resolution_rank(path_text: str) -> Tuple[int, str]:
    lower = path_text.lower().replace("\\", "/")
    if "/r10m/" in lower:
        rank = 0
    elif "/r20m/" in lower:
        rank = 1
    elif "/r60m/" in lower:
        rank = 2
    else:
        rank = 9
    return rank, lower


def discover_sen2cor_safe(safe: Path, years: Sequence[int], dates: Sequence[str], all_dates: bool) -> Optional[Scene]:
    fallback_date = parse_date_from_path(safe)
    metadata_file = find_first_existing(["MTD_MSIL2A.xml", "MTD_*.xml"], safe)
    if metadata_file is not None:
        date, quant, nodata, offsets = parse_sen2cor_metadata_file(metadata_file, fallback_date)
    else:
        date, quant, nodata, offsets = fallback_date, 10000.0, 0.0, {}

    if date is None or not include_date(date, years, dates, all_dates):
        return None

    candidates: Dict[str, List[Path]] = defaultdict(list)
    for jp2 in safe.rglob("*.jp2"):
        match = S2_BAND_PATTERN.search(jp2.name)
        if match is not None:
            candidates[normalize_s2_band(match.group(1))].append(jp2)

    bands: Dict[str, BandSource] = {}
    for band, files in sorted(candidates.items(), key=lambda item: band_sort_key(item[0])):
        files.sort(key=lambda p: sen2cor_resolution_rank(str(p)))
        source = files[0]
        bands[band] = BandSource(
            band=band,
            open_path=str(source),
            display_path=str(source),
            scale=1.0 / quant,
            offset=offsets.get(band, 0.0),
            nodata=nodata,
        )

    if not bands:
        return None

    return Scene("Sen2Cor", date, safe.name, str(safe), bands)


def discover_sen2cor_zip(zip_path: Path, years: Sequence[int], dates: Sequence[str], all_dates: bool) -> Optional[Scene]:
    fallback_date = parse_date_from_path(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    metadata_members = [
        member
        for member in members
        if member.endswith("/MTD_MSIL2A.xml") and len(PurePosixPath(member).parts) == 2
    ]
    metadata_member = metadata_members[0] if metadata_members else None
    if metadata_member is not None:
        date, quant, nodata, offsets = parse_sen2cor_metadata_zip(zip_path, metadata_member, fallback_date)
    else:
        date, quant, nodata, offsets = fallback_date, 10000.0, 0.0, {}

    if date is None or not include_date(date, years, dates, all_dates):
        return None

    candidates: Dict[str, List[str]] = defaultdict(list)
    for member in members:
        if not member.lower().endswith(".jp2"):
            continue
        name = PurePosixPath(member).name
        match = S2_BAND_PATTERN.search(name)
        if match is not None:
            candidates[normalize_s2_band(match.group(1))].append(member)

    bands: Dict[str, BandSource] = {}
    zip_vsi_path = f"/vsizip/{zip_path.resolve().as_posix()}"
    for band, files in sorted(candidates.items(), key=lambda item: band_sort_key(item[0])):
        files.sort(key=sen2cor_resolution_rank)
        member = files[0]
        bands[band] = BandSource(
            band=band,
            open_path=f"{zip_vsi_path}/{member}",
            display_path=f"{zip_path}!{member}",
            scale=1.0 / quant,
            offset=offsets.get(band, 0.0),
            nodata=nodata,
        )

    if not bands:
        return None

    safe_name = PurePosixPath(members[0]).parts[0] if members else zip_path.stem
    return Scene("Sen2Cor", date, safe_name, str(zip_path), bands)


def discover_sen2cor(roots: Sequence[Path], years: Sequence[int], dates: Sequence[str], all_dates: bool) -> List[Scene]:
    scenes: List[Scene] = []
    for product in find_sen2cor_products(roots):
        try:
            if product.is_file() and product.suffix.lower() == ".zip":
                scene = discover_sen2cor_zip(product, years, dates, all_dates)
            else:
                scene = discover_sen2cor_safe(product, years, dates, all_dates)
        except Exception as exc:
            print(f"Skipping unreadable Sen2Cor product {product}: {exc}", file=sys.stderr)
            continue
        if scene is not None:
            scenes.append(scene)
    return scenes


def acolite_wavelength_to_band(wavelength: float) -> Optional[str]:
    band, delta = min(
        ((band, abs(center - wavelength)) for band, center in ACOLITE_BAND_CENTRES.items()),
        key=lambda item: item[1],
    )
    max_delta = 120.0 if wavelength > 1000 else 20.0
    return band if delta <= max_delta else None


def parse_acolite_rhos_name(text: str) -> Optional[Tuple[str, str]]:
    match = ACOLITE_RHOS_PATTERN.search(text)
    if match is None:
        return None
    wavelength = float(match.group("wavelength"))
    band = acolite_wavelength_to_band(wavelength)
    if band is None:
        return None
    return band, match.group(1)


def acolite_scene_key_from_name(name: str, rhos_name: str) -> str:
    index = name.lower().find(rhos_name.lower())
    return name[:index].rstrip("_-") if index >= 0 else name


def list_reflectance_subdatasets(nc_file: Path) -> Dict[str, BandSource]:
    import rasterio

    bands: Dict[str, BandSource] = {}
    with rasterio.open(nc_file) as dataset:
        subdatasets = list(dataset.subdatasets)

    for subdataset in subdatasets:
        parsed = parse_acolite_rhos_name(subdataset)
        if parsed is None:
            continue
        band, _rhos_name = parsed
        bands[band] = BandSource(band=band, open_path=subdataset, display_path=str(nc_file))
    return bands


def discover_acolite(root: Path, years: Sequence[int], dates: Sequence[str], all_dates: bool) -> List[Scene]:
    if not root.exists():
        return []

    scenes: List[Scene] = []
    grouped_tifs: Dict[Tuple[str, str], Dict[str, BandSource]] = defaultdict(dict)
    grouped_sources: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    tif_files = sorted(
        list(root.rglob("*.tif")) + list(root.rglob("*.tiff")),
        key=lambda p: str(p).lower(),
    )
    for tif in tif_files:
        parsed = parse_acolite_rhos_name(tif.stem)
        if parsed is None:
            continue
        band, rhos_name = parsed
        date = parse_date_from_path(tif)
        if date is None or not include_date(date, years, dates, all_dates):
            continue
        scene_key = acolite_scene_key_from_name(tif.stem, rhos_name)
        key = (date, scene_key)
        grouped_tifs[key][band] = BandSource(band=band, open_path=str(tif), display_path=str(tif))
        grouped_sources[key].append(str(tif))

    for (date, scene_key), bands in sorted(grouped_tifs.items()):
        sources = sorted(grouped_sources[(date, scene_key)])
        source = os.path.commonpath(sources) if len(sources) > 1 else sources[0]
        scenes.append(Scene("ACOLITE", date, scene_key, source, bands))

    # Use NetCDF only for dates where no GeoTIFF reflectance stack was found.
    tif_scene_dates = {date for date, _scene_key in grouped_tifs}
    for nc_file in sorted(root.rglob("*.nc"), key=lambda p: str(p).lower()):
        date = parse_date_from_path(nc_file)
        if date is None or date in tif_scene_dates or not include_date(date, years, dates, all_dates):
            continue
        try:
            bands = list_reflectance_subdatasets(nc_file)
        except Exception as exc:
            print(f"Skipping unreadable ACOLITE NetCDF {nc_file}: {exc}", file=sys.stderr)
            continue
        if bands:
            scenes.append(Scene("ACOLITE", date, nc_file.stem, str(nc_file), bands))

    return scenes


def read_aois(shapefile: Path, aoi_field: str):
    import fiona

    if not shapefile.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {shapefile}")

    aois = []
    with fiona.open(shapefile) as src:
        crs = src.crs_wkt or src.crs
        for feature in src:
            properties = dict(feature["properties"])
            if aoi_field not in properties:
                raise KeyError(f"AOI field '{aoi_field}' not found in {shapefile}")
            aois.append({"name": properties[aoi_field], "geometry": dict(feature["geometry"]), "crs": crs})
    return aois


def choose_reference_band(scene: Scene) -> BandSource:
    for band in ("B02", "B03", "B04", "B08", "B01"):
        if band in scene.bands:
            return scene.bands[band]
    return scene.bands[sorted(scene.bands.keys(), key=band_sort_key)[0]]


def clean_value(value, source: BandSource) -> float:
    if np.ma.is_masked(value):
        return math.nan
    value = float(np.asarray(value).reshape(-1)[0])
    if source.nodata is not None and value == source.nodata:
        return math.nan
    if not math.isfinite(value):
        return math.nan
    return (value + source.offset) * source.scale


def extract_scene_rows(scene: Scene, aois, all_touched: bool) -> Iterator[Dict[str, object]]:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.errors import WindowError
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.transform import xy
    from rasterio.warp import transform, transform_geom

    reference = choose_reference_band(scene)
    with rasterio.open(reference.open_path) as ref_ds:
        ref_crs = ref_ds.crs
        for aoi in aois:
            geom = aoi["geometry"]
            aoi_crs = CRS.from_user_input(aoi["crs"]) if aoi["crs"] else None
            if aoi_crs and ref_crs and aoi_crs != ref_crs:
                geom = transform_geom(aoi_crs, ref_crs, geom)
            try:
                window = geometry_window(ref_ds, [geom])
            except WindowError:
                continue

            mask = geometry_mask(
                [geom],
                out_shape=(int(window.height), int(window.width)),
                transform=ref_ds.window_transform(window),
                invert=True,
                all_touched=all_touched,
            )
            rows, cols = np.nonzero(mask)
            if len(rows) == 0:
                continue

            abs_rows = rows + int(window.row_off)
            abs_cols = cols + int(window.col_off)
            xs, ys = xy(ref_ds.transform, abs_rows, abs_cols, offset="center")
            xs = list(xs)
            ys = list(ys)
            points_ref = list(zip(xs, ys))

            band_values: Dict[str, List[float]] = {}
            for band, source in scene.bands.items():
                with rasterio.open(source.open_path) as band_ds:
                    points = points_ref
                    if ref_crs and band_ds.crs and band_ds.crs != ref_crs:
                        bx, by = transform(ref_crs, band_ds.crs, xs, ys)
                        points = list(zip(bx, by))
                    band_values[band] = [
                        clean_value(sample[0], source)
                        for sample in band_ds.sample(points, masked=True)
                    ]

            for pixel_index, (row, col, x_coord, y_coord) in enumerate(zip(abs_rows, abs_cols, xs, ys)):
                out = {
                    "date": scene.date,
                    "AOI": aoi["name"],
                    "atmospheric_correction": scene.correction,
                    "scene_id": scene.scene_id,
                    "source": scene.source,
                    "pixel_id": pixel_index,
                    "row": int(row),
                    "col": int(col),
                    "x": x_coord,
                    "y": y_coord,
                }
                for band in scene.bands:
                    out[band] = band_values[band][pixel_index]
                yield out


def pixel_key(row: Dict[str, object]) -> Tuple[object, float, float]:
    return row["AOI"], round(float(row["x"]), 6), round(float(row["y"]), 6)


def write_outputs(
    scenes: Sequence[Scene],
    aois,
    output_root: Path,
    all_touched: bool,
    drop_overlap_duplicates: bool,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    grouped: Dict[Tuple[str, str], List[Scene]] = defaultdict(list)
    for scene in scenes:
        grouped[(scene.correction, scene.date)].append(scene)

    for (correction, date), date_scenes in sorted(grouped.items()):
        output_dir = output_root / correction
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{date}.csv"
        band_columns = sorted({band for scene in date_scenes for band in scene.bands}, key=band_sort_key)
        fieldnames = METADATA_COLUMNS + band_columns

        row_count = 0
        skipped_duplicates = 0
        seen_pixels = set()
        with output_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for scene in date_scenes:
                for row in extract_scene_rows(scene, aois, all_touched=all_touched):
                    if drop_overlap_duplicates:
                        key = pixel_key(row)
                        if key in seen_pixels:
                            skipped_duplicates += 1
                            continue
                        seen_pixels.add(key)
                    writer.writerow(row)
                    row_count += 1
        counts[str(output_file)] = row_count
        duplicate_message = f" ({skipped_duplicates} overlap duplicates skipped)" if skipped_duplicates else ""
        print(f"Wrote {row_count} rows to {output_file}{duplicate_message}")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Sentinel-2 reflectance values for AOI polygons.")
    parser.add_argument(
        "--sen2cor-root",
        dest="sen2cor_roots",
        action="append",
        help=(
            "Sen2Cor SAFE directory, zip file, or directory containing products. "
            "Can be repeated. Defaults to Data/S2/Sen2Cor and Data/S2."
        ),
    )
    parser.add_argument("--acolite-root", default="Data/S2/ACOLITE")
    parser.add_argument("--shapefile", default="Data/shp/TimeSeries_Polygons_Matchup.shp")
    parser.add_argument("--output-root", default="Data/Atm_Compare_AOI")
    parser.add_argument("--aoi-field", default="AOI")
    parser.add_argument("--year", dest="years", type=int, action="append")
    parser.add_argument("--date", dest="dates", action="append")
    parser.add_argument("--all-dates", action="store_true", help="Ignore --year/--date filters.")
    parser.add_argument("--all-touched", action="store_true", help="Include all pixels touched by each polygon.")
    parser.add_argument(
        "--keep-overlap-duplicates",
        action="store_true",
        help="Keep duplicate AOI/x/y rows when adjacent Sentinel-2 tiles overlap.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    require_dependencies()

    years = args.years if args.years is not None else [2025]
    dates = [normalize_date(date) for date in (args.dates if args.dates is not None else ["2026-03-27"])]

    sen2cor_roots = args.sen2cor_roots if args.sen2cor_roots is not None else ["Data/S2/Sen2Cor", "Data/S2"]
    sen2cor_roots = [resolve_path(root) for root in sen2cor_roots]
    acolite_root = resolve_path(args.acolite_root)
    shapefile = resolve_path(args.shapefile)
    output_root = resolve_path(args.output_root)

    aois = read_aois(shapefile, args.aoi_field)
    print(f"Loaded {len(aois)} AOIs from {shapefile}")

    scenes = []
    scenes.extend(discover_sen2cor(sen2cor_roots, years, dates, args.all_dates))
    scenes.extend(discover_acolite(acolite_root, years, dates, args.all_dates))
    print(f"Discovered {len(scenes)} scenes matching filters.")

    if not scenes:
        print("No matching scenes found.")
        return 1

    write_outputs(
        scenes,
        aois,
        output_root,
        all_touched=args.all_touched,
        drop_overlap_duplicates=not args.keep_overlap_duplicates,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
