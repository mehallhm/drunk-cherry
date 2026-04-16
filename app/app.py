import os

os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path

import gpxpy
import keras
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (match src/scripts/image_gen.py)
# ---------------------------------------------------------------------------
IMG_SIZE = 256
PADDING_PX = 24
LINE_WIDTH = 4
DOT_RADIUS = 3

MODEL_PATH = Path(__file__).resolve().parent / "model" / "trail_cnn_model.keras"
EXAMPLES_DIR = Path(__file__).resolve().parent / "gpx_examples"

# The LabelEncoder in training sorts classes alphabetically.
# From trail_helpers.DIFFICULTY_MAP the 4 consolidated classes are:
CLASS_LABELS = ["Difficult", "Easy", "Intermediate", "Intermediate/Difficult"]

# Friendly display names
DISPLAY_LABELS = {
    "Difficult": "Difficult",
    "Easy": "Easy",
    "Intermediate": "Intermediate",
    "Intermediate/Difficult": "Intermediate / Difficult",
}


# ---------------------------------------------------------------------------
# GPX parsing
# ---------------------------------------------------------------------------
def parse_gpx(file_bytes: bytes) -> pd.DataFrame:
    """Return a DataFrame with columns: latitude, longitude, elevation."""
    gpx = gpxpy.parse(file_bytes.decode("utf-8"))

    points: list[dict] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append(
                    {
                        "latitude": pt.latitude,
                        "longitude": pt.longitude,
                        "elevation": pt.elevation if pt.elevation is not None else 0.0,
                    }
                )

    # Fall back to waypoints / route points when there are no track points
    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append(
                    {
                        "latitude": pt.latitude,
                        "longitude": pt.longitude,
                        "elevation": pt.elevation if pt.elevation is not None else 0.0,
                    }
                )

    if not points:
        for pt in gpx.waypoints:
            points.append(
                {
                    "latitude": pt.latitude,
                    "longitude": pt.longitude,
                    "elevation": pt.elevation if pt.elevation is not None else 0.0,
                }
            )

    if len(points) < 2:
        raise ValueError(
            "GPX file must contain at least 2 points (track, route, or waypoints)."
        )

    return pd.DataFrame(points)


# ---------------------------------------------------------------------------
# Compute extra features expected by the CNN
# ---------------------------------------------------------------------------
def compute_extra_features(df: pd.DataFrame) -> np.ndarray:
    """Return array of [elevation_gain, elevation_loss, average_grade, max_grade]."""
    elevs = df["elevation"].values.astype(float)
    lats = np.radians(df["latitude"].values.astype(float))
    lons = np.radians(df["longitude"].values.astype(float))

    diffs = np.diff(elevs)
    elevation_gain = float(np.sum(diffs[diffs > 0]))
    elevation_loss = float(np.abs(np.sum(diffs[diffs < 0])))

    # Haversine horizontal distances (metres)
    dlat = np.diff(lats)
    dlon = np.diff(lons)
    a = np.sin(dlat / 2) ** 2 + np.cos(lats[:-1]) * np.cos(lats[1:]) * np.sin(dlon / 2) ** 2
    horiz = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    safe_horiz = np.where(horiz > 0.01, horiz, 0.01)  # avoid division by zero
    grades = np.abs(diffs) / safe_horiz * 100.0

    average_grade = float(np.mean(grades))
    max_grade = float(np.max(grades))

    return np.array(
        [elevation_gain, elevation_loss, average_grade, max_grade],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Image generation (grayscale, per-trail elevation scaling)
# Adapted from src/scripts/image_gen.py
# ---------------------------------------------------------------------------
def _blend(c0: tuple, c1: tuple, t: float) -> tuple:
    return (
        int(c0[0] + t * (c1[0] - c0[0])),
        int(c0[1] + t * (c1[1] - c0[1])),
        int(c0[2] + t * (c1[2] - c0[2])),
    )


def _draw_line(pixels: np.ndarray, x0, y0, c0, x1, y1, c1, thickness):
    h, w = pixels.shape[:2]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    total = max(1, dx + dy)
    travelled = 0
    cx, cy = x0, y0
    r = thickness - 1
    while True:
        t = travelled / total
        gray = int(c0[0] + t * (c1[0] - c0[0]))
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                px, py = cx + ox, cy + oy
                if 0 <= px < w and 0 <= py < h:
                    pixels[py, px] = gray
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


def _draw_dot(pixels: np.ndarray, cx, cy, radius, gray_val):
    h, w = pixels.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    pixels[py, px] = gray_val


def generate_image(df: pd.DataFrame) -> Image.Image:
    """Generate a 256x256 grayscale trail image from a DataFrame of points."""
    lats = df["latitude"].values.astype(float)
    lons = df["longitude"].values.astype(float)
    elevs = df["elevation"].values.astype(float)

    lat_range = max(lats.max() - lats.min(), 1e-9)
    lon_range = max(lons.max() - lons.min(), 1e-9)
    drawable = IMG_SIZE - 2 * PADDING_PX

    def to_px(lat, lon):
        x = int(PADDING_PX + (lon - lons.min()) / lon_range * drawable)
        y = int(PADDING_PX + (1.0 - (lat - lats.min()) / lat_range) * drawable)
        return x, y

    elev_min, elev_max = float(elevs.min()), float(elevs.max())
    elev_range = max(elev_max - elev_min, 1e-9)

    def norm(e):
        return (e - elev_min) / elev_range

    avg_intensity = int(norm(float(np.mean(elevs))) * 255)
    pixels = np.full((IMG_SIZE, IMG_SIZE), avg_intensity, dtype=np.uint8)

    coords = [to_px(lat, lon) for lat, lon in zip(lats, lons)]
    grays = [(int(norm(e) * 255),) * 3 for e in elevs]

    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        _draw_line(pixels, x0, y0, grays[i], x1, y1, grays[i + 1], LINE_WIDTH)

    for (px, py), g in zip(coords, grays):
        _draw_dot(pixels, px, py, DOT_RADIUS, g[0])

    # start / end markers
    _draw_dot(pixels, coords[0][0], coords[0][1], DOT_RADIUS * 2, 255)
    _draw_dot(pixels, coords[-1][0], coords[-1][1], DOT_RADIUS * 2, 128)

    return Image.fromarray(pixels, mode="L")


# ---------------------------------------------------------------------------
# Model loading & prediction
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> keras.Model:
    return keras.models.load_model(MODEL_PATH)


def predict(model: keras.Model, image: Image.Image, features: np.ndarray) -> tuple[str, np.ndarray]:
    img_arr = np.array(image, dtype=np.float32) / 255.0
    img_arr = img_arr[np.newaxis, ..., np.newaxis]  # (1, 256, 256, 1)

    feat = features[np.newaxis, :]  # (1, 4)

    # Normalize features to [0, 1] using approximate dataset ranges.
    # These were estimated from the all_trails dataset; adjust if needed.
    feat_min = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    feat_max = np.array([5000.0, 5000.0, 50.0, 150.0], dtype=np.float32)
    feat_range = np.where(feat_max - feat_min > 1e-9, feat_max - feat_min, 1.0)
    feat = (feat - feat_min) / feat_range
    feat = np.clip(feat, 0.0, 1.0)

    probs = model.predict([img_arr, feat], verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_label = CLASS_LABELS[predicted_idx]
    return predicted_label, probs


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def landing_page():
    st.header("About")
    st.write(
        """
Trail rating remains a difficult and inherently human process. Across a variety of outdoor activities (e.g. hiking, cycling, and skiing) difficulty ratings rely on relative systems defined and applied locally, preventing trail users from comparing difficulties between areas. At worst, this inconsistency poses a safety concern: a rider or hiker who performs competently at a given rating in one region, may encounter dramatically more challenging terrain under the same rating elsewhere, leading to potential injury. Adopting a more algorithmic approach, driven by quantifiable trail features and standardized across regions, would allow users to access appropriate difficulty information and make informed decisions that support both safety and activity enjoyment.
The space of categorizing mountain biking (MTB) trails specifically is of interest. The International Mountain Bicycling Association (IMBA) provides a widely referenced rating scale, yet its application is ultimately left to local trail builders and land maintainers, introducing the subjectivity and regional inconsistencies observed in other disciplines. Further, compared to hiking or skiing, relatively little prior work has explored data-driven approaches to MTB trail classification, leaving the problem largely unaddressed in the literature.
A promising data source for this task is the GPX file format. GPX files are routinely recorded by riders using consumer GPS devices and encode georeferenced sequences of latitude, longitude, and elevation at high temporal resolution. From these traces, features relevant to perceived difficulty, such as total elevation gain, grade, elevation variability, and descent profiles, can be extracted without requiring specialized survey equipment. The ubiquity of GPX data, combined with platforms that aggregate user-recorded rides alongside community-assigned ratings, creates an opportunity to build labeled datasets at scale.
This work presents a two-pronged modeling approach to automated MTB trail difficulty classification. A convolutional neural network (CNN) is trained to learn spatial patterns directly from processed GPX representations, while a Bayesian model offers a complementary probabilistic framework that quantifies uncertainty in its predictions, especially valuable given the subjectivity inherent in trail ratings. To demonstrate the practical applicability, the resulting models are integrated into a web application that allows a user to upload a GPX file and receive a predicted difficulty rating.
"""
    )
    st.subheader("How it works")
    st.write(
        "1. Upload a GPX file or select one of the bundled examples.\n"
        "2. The trail is rendered as a grayscale elevation heatmap.\n"
        "3. A CNN model predicts the trail difficulty."
    )


def predictor_page():
    # ---- discover bundled example GPX files ----
    examples: dict[str, Path] = {}
    if EXAMPLES_DIR.is_dir():
        for f in sorted(EXAMPLES_DIR.iterdir()):
            if f.suffix.lower() == ".gpx":
                examples[f.stem] = f

    # ---- input selection ----
    option_names = ["Upload my own file"] + list(examples.keys())
    choice = st.selectbox("Choose a trail", option_names)

    raw: bytes | None = None

    if choice == "Upload my own file":
        uploaded = st.file_uploader("Upload a GPX file", type=["gpx"])
        if uploaded is not None:
            raw = uploaded.read()
    else:
        raw = examples[choice].read_bytes()

    if raw is None:
        st.info("Select an example trail or upload your own GPX file.")
        return

    try:
        df = parse_gpx(raw)
    except Exception as exc:
        st.error(f"Failed to parse GPX file: {exc}")
        return

    st.success(f"Parsed {len(df)} GPS points.")

    # ---- compute everything up front ----
    image = generate_image(df)
    features = compute_extra_features(df)

    # ---- two-column layout ----
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Trail Visualization")
        st.image(image, caption="Elevation heatmap (grayscale)", width="stretch")

    with col_right:
        # ---- prediction ----
        if not MODEL_PATH.exists():
            st.warning(
                f"Model not found at `{MODEL_PATH}`. "
                "Copy `src/models/trail_cnn_model.keras` into `app/model/`."
            )
            return

        model = load_model()
        label, probs = predict(model, image, features)

        st.subheader("Predicted Difficulty")
        st.markdown(f"### {DISPLAY_LABELS[label]}")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            {"Difficulty": [DISPLAY_LABELS[c] for c in CLASS_LABELS], "Probability": probs}
        )
        st.bar_chart(prob_df.set_index("Difficulty"))

        st.subheader("Trail Stats")
        st.metric("Elevation Gain", f"{features[0]:.1f} m")
        st.metric("Elevation Loss", f"{features[1]:.1f} m")
        st.metric("Average Grade", f"{features[2]:.2f}%")
        st.metric("Max Grade", f"{features[3]:.2f}%")


def main():
    st.set_page_config(page_title="Trail Difficulty Predictor", layout="wide")
    st.title("Trail Difficulty Predictor")

    tab_home, tab_predict = st.tabs(["Home", "Predict"])

    with tab_home:
        landing_page()

    with tab_predict:
        predictor_page()


if __name__ == "__main__":
    main()
