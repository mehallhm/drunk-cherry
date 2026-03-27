# Michael Mehall & Paolo Lanaro - Image Generator from bike trail data

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = 256  # output image width adn height in pixels
PADDING_PX = 24  # blank border padding around the trail
LINE_WIDTH = 4  # pixel thickness (radius) of the trail line
DOT_RADIUS = 3  # radius of GPS point dots drawn on top of lines

elevation_to_RGB_mapping = [
    (0.00, np.array([0, 0, 180], dtype=float)),
    (0.25, np.array([0, 200, 200], dtype=float)),
    (0.50, np.array([0, 200, 0], dtype=float)),
    (0.75, np.array([220, 220, 0], dtype=float)),
    (1.00, np.array([220, 0, 0], dtype=float)),
]

known_diffs = {
    "Easy",
    "Easy/Intermediate",
    "Intermediate",
    "Difficult",
    "Intermediate/Difficult",
    "Very Difficult",
}


# maps normalized elevation value to an RGB color
def elevation_to_color(norm_value: float) -> tuple:
    # just a safeguard in case the normalization somehow didn't work
    value = float(np.clip(norm_value, 0.0, 1.0))

    for i in range(len(elevation_to_RGB_mapping) - 1):
        thresh0, color0 = elevation_to_RGB_mapping[i]
        thresh1, color1 = elevation_to_RGB_mapping[i + 1]
        if thresh0 <= value <= thresh1:
            alpha = (value - thresh0) / (thresh1 - thresh0)
            rgb = color0 + alpha * (color1 - color0)
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # this function WILL necessarily return above but i'm getting an lsp error so:
    return (0, 0, 0)


# blends two colors by a ratio
def blend_colors(color0: tuple, color1: tuple, ratio: float) -> tuple:
    return (
        int(color0[0] + ratio * (color1[0] - color0[0])),
        int(color0[1] + ratio * (color1[1] - color0[1])),
        int(color0[2] + ratio * (color1[2] - color0[2])),
    )


# Shoutout señor Bresenham: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
def draw_line_segment(
    pixels: np.ndarray,
    x0: int,
    y0: int,
    color0: tuple,
    x1: int,
    y1: int,
    color1: tuple,
    thickness: int = 2,
) -> None:
    h, w = pixels.shape[:2]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    # we take the max of 1 and dx + dy because we don't want a possible division by zero 7 lines down from this line
    total_dist = max(1, dx + dy)  # L1 norm
    travelled = 0

    cx, cy = x0, y0

    while True:
        progress = travelled / total_dist
        spine_color = blend_colors(color0, color1, progress)

        # draw square of size (2*thickness-1) around the spine pixel for thickness
        radius = thickness - 1
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                px, py = cx + ox, cy + oy
                if 0 <= px < w and 0 <= py < h:
                    pixels[py, px] = spine_color

        if cx == x1 and cy == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
            travelled += 1
        if e2 < dx:
            err += dx
            cy += sy
            travelled += 1


def draw_dot(pixels: np.ndarray, cx: int, cy: int, radius: int, color: tuple) -> None:
    height, width = pixels.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius**2:
                px, py = cx + dx, cy + dy
                if 0 <= px < width and 0 <= py < height:
                    pixels[py, px] = color


# create a single image for trail_id with trail_df
# scaled to fill the canvas and elevation is normzlied against global min and max elevations.
# This means a flat trail is mostly blue but a high elevation gain/loss trail spans blue to red
def generate_trail_image(
    trail_df: pd.DataFrame,
    trail_id,
    output_dir: Path,
    global_elev_min: float,
    global_elev_max: float,
) -> Path:
    latitudes = trail_df["latitude"].values.astype(float)
    longitudes = trail_df["longitude"].values.astype(float)
    elevs = trail_df["elevation"].values.astype(float)

    # normalize the trail so it fills the canvas
    # I'm making sure that the lat and lon ranges can never be 0 so that we don't get a divide by 0!
    lat_range = max(latitudes.max() - latitudes.min(), 1e-9)
    lon_range = max(longitudes.max() - longitudes.min(), 1e-9)
    drawable = IMG_SIZE - 2 * PADDING_PX

    # normalize the actual GPS points in the canvas. This is what "lays" them out correctly in the image
    def to_coordinates(latitude, longitude):
        px = int(PADDING_PX + (longitude - longitudes.min()) / lon_range * drawable)
        py = int(
            PADDING_PX + (1.0 - (latitude - latitudes.min()) / lat_range) * drawable
        )
        return px, py

    global_elevation_range = global_elev_max - global_elev_min

    def normalize_elevation(elevation):
        return (elevation - global_elev_min) / global_elevation_range

    # create image canvas
    pixels = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    pixels[:] = (0, 0, 0)  # background

    # pixel positions and colors for all GPS point
    coords = [to_coordinates(lat, lon) for lat, lon in zip(latitudes, longitudes)]
    colors = [elevation_to_color(normalize_elevation(elevation)) for elevation in elevs]

    # connected line segments first
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        draw_line_segment(
            pixels, x0, y0, colors[i], x1, y1, colors[i + 1], thickness=LINE_WIDTH
        )

    # draw dots at each GPS point on top of the lines
    for (px, py), color in zip(coords, colors):
        draw_dot(pixels, px, py, DOT_RADIUS, color)

    start_color = (255, 255, 255)
    end_color = (255, 0, 255)

    # create starting (white) and ending (purple) markers
    draw_dot(pixels, coords[0][0], coords[0][1], DOT_RADIUS * 2, start_color)
    draw_dot(pixels, coords[-1][0], coords[-1][1], DOT_RADIUS * 2, end_color)

    img = Image.fromarray(pixels, mode="RGB")

    difficulty = (
        trail_df["difficulty"].iloc[0]
        if "difficulty" in trail_df.columns
        else "unknown"
    )
    safe_diff = "".join(c if c.isalnum() else "_" for c in str(difficulty))
    out_path = output_dir / f"trail_{trail_id}_{safe_diff}.png"
    img.save(str(out_path))
    return out_path


def process_single_trail(
    df: pd.DataFrame,
    trail_id,
    output_dir: Path,
    global_elev_min: float,
    global_elev_max: float,
) -> None:
    subset = df[df["trail_id"] == trail_id]
    if subset.empty:
        print(f"[ERROR] trail_id {trail_id} not found in the dataset.")
        sys.exit(1)
    out = generate_trail_image(
        subset, trail_id, output_dir, global_elev_min, global_elev_max
    )
    print(f"Saved: {out}")


def worker(args):
    trail_id, group_df, output_dir, elevation_min, elevation_max = args
    try:
        out = generate_trail_image(
            group_df, trail_id, output_dir, elevation_min, elevation_max
        )
        return trail_id, out
    except Exception as exc:
        return trail_id, exc


def process_all_trails(
    df: pd.DataFrame,
    output_dir: Path,
    num_threads: int,
    global_elev_min: float,
    global_elev_max: float,
) -> None:
    groups = list(df.groupby("trail_id"))
    total = len(groups)
    failed_images = set()

    print(f"Processing {total} trails with {num_threads} thread(s)...")

    work_items = [
        (trail_id, group_df, output_dir, global_elev_min, global_elev_max)
        for trail_id, group_df in groups
    ]

    if num_threads == 1:
        for i, item in enumerate(work_items, 1):
            _, result = worker(item)
            if isinstance(result, Exception):
                failed_images.add(item[0])
            elif i % 500 == 0 or i == total:
                print(f"  {i}/{total} done -- last completed image: {result}")
            if i == 2000:
                print(" === Early finishing after 200 images ===")
                break
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(worker, item): item[0] for item in work_items}
            for future in as_completed(futures):
                trail_id, result = future.result()
                done += 1
                if isinstance(result, Exception):
                    failed_images.add(trail_id)
                elif done % 1000 == 0 or done == total:
                    print(f"  {done}/{total} done -- last completed image: {result}")

    print("All trails processed.")

    if len(failed_images):
        print(f"[WARN] The following trail ids failed to convert: {failed_images}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2D elevation heatmap images for bike trail GPS data"
            "Example Usage:\n"
            "  python trail_heatmap.py --input trails.csv --trail_id 12376\n"
            "  python trail_heatmap.py --input trails.csv --all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        metavar="CSV_PATH",
        help="Path to the input CSV file",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--trail_id",
        type=int,
        metavar="ID",
        help="Process a single trail by its trail_id",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Process every trail in the CSV",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("trail_data_images"),
        metavar="OUTPUT_DIR",
        help="Directory where images are saved (default: ./trail_data_images/)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of threads for --all mode (default: 1). Pass 0 to use all but one available CPU cores."
        ),
    )

    return parser


def filter_bad_difficulties(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    bad_ids = {
        trail_id
        for trail_id, group in df.groupby("trail_id")
        if bool(group["difficulty"].isnull().any())
        or not bool(group["difficulty"].dropna().map(lambda d: d in known_diffs).all())
    }

    return df[~df["trail_id"].isin(list(bad_ids))], bad_ids


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path: Path = args.input
    if not csv_path.is_file():
        parser.error(f"Input file not found: {csv_path}")

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, index_col=0)
    print("Succesfully read CSV")
    print(f"Loaded {len(df):,} rows, {df['trail_id'].nunique():,} unique trails.")

    df, bad_diff_ids = filter_bad_difficulties(df)
    print(f"bad_diff_ids: {bad_diff_ids}")

    # computes global elevation bounds.
    # trails that don't climb much stay blue but trains that climb a lot of elevatoin go from blue to red
    global_elev_min = float(min(df["elevation"].values))
    global_elev_max = float(max(df["elevation"].values))
    print(f"Global elevation range: {global_elev_min:.1f} m -> {global_elev_max:.1f} m")

    if args.trail_id is not None:
        process_single_trail(
            df, args.trail_id, output_dir, global_elev_min, global_elev_max
        )
    else:
        # either use specified number of threads, or one minus logical CPU count
        n_threads = args.threads if args.threads > 0 else ((os.cpu_count() or 2) - 1)
        process_all_trails(df, output_dir, n_threads, global_elev_min, global_elev_max)


if __name__ == "__main__":
    main()
