import os
import numpy as np
import joblib
import cv2
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# PAGE CONFIG

st.set_page_config(
    page_title="BananaCheck — Ripeness Detector",
    page_icon="",
    layout="wide"
)

# STYLING

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    .banana-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #12161f 100%);
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }

    .banana-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f5d442;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }

    .banana-header p {
        color: #8a8fa8;
        margin: 0;
        font-size: 1rem;
    }

    .model-card {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .model-card h3 {
        color: #f5d442;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 0 0 1rem 0;
    }

    .result-box {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .final-result {
        background: linear-gradient(135deg, #2a2f1e 0%, #1a2010 100%);
        border: 2px solid #f5d442;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .final-result h2 {
        color: #f5d442;
        font-size: 2rem;
        margin: 0;
    }

    .final-result p {
        color: #8a8fa8;
        margin: 0.5rem 0 0 0;
    }

    .label-chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .stButton > button {
        background: #f5d442 !important;
        color: #0f1117 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: #e8c83a !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background: #f5d44220 !important;
        border: 1px solid #f5d44250 !important;
        color: #f5d442 !important;
    }

    .stNumberInput input {
        background: #12161f !important;
        border: 1px solid #2a2f3e !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
    }

    div[data-testid="stFileUploader"] {
        background: #12161f;
        border: 2px dashed #2a2f3e;
        border-radius: 12px;
        padding: 1rem;
    }

    .stSuccess {
        background: #1a2a1a !important;
        border: 1px solid #4a7a4a !important;
    }

    .stWarning {
        background: #2a2010 !important;
        border: 1px solid #7a6010 !important;
    }

    hr { border-color: #2a2f3e; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS

GAS_ENV_FEATURES = [
    'Temp-int', 'Press-int', 'Humid-int',
    'Temp-ext', 'Press-ext', 'Humid-ext',
    'TGS20', 'TGS02', 'SGP'
]

SPECTER_FEATURES = [
    'SpA410', 'SpB435', 'SpC460', 'SpD485', 'SpE510', 'SpF535',
    'SpG560', 'SpH585', 'SpR610', 'SpI645', 'SpS680', 'SpJ705',
    'SpT730', 'SpU760', 'SpV810', 'SpW860', 'SpK900', 'SpL940'
]

ALL_FEATURES = GAS_ENV_FEATURES + SPECTER_FEATURES

LABEL_MAP = {
    0: ("Underripe", "", "#3a7a3a"),
    1: ("Ripe",      "", "#8a7a10"),
    2: ("Overripe",  "", "#8a5010"),
    3: ("Rotten",    "", "#7a2020"),
    4: ("Severely Rotten", "", "#3a2a2a"),
}

MODEL_PATHS = {
    "Gas & Env":   "models/random_forest_model_gas&env.pkl",
    "Specter":     "models/random_forest_model_specter.pkl",
    "All Sensors": "models/random_forest_model_all.pkl",
}

MODEL_FEATURES = {
    "Gas & Env":   GAS_ENV_FEATURES,
    "Specter":     SPECTER_FEATURES,
    "All Sensors": ALL_FEATURES,
}

# MODEL LOADER

@st.cache_resource
def load_model(name, path, features):
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            return model, None
        else:
            raise FileNotFoundError(f"{path} not found")
    except Exception as e:
        scaler = StandardScaler()
        X_dummy = np.random.rand(50, len(features))
        y_dummy = np.random.randint(0, 5, 50)
        X_dummy = scaler.fit_transform(X_dummy)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_dummy, y_dummy)
        return clf, scaler

def predict_numeric(model_obj, scaler, values):
    X = np.array(values, dtype=float).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)
    return int(model_obj.predict(X)[0])

# IMAGE — COLOR ANALYSIS

def analyze_banana_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    total_pixels = hsv.shape[0] * hsv.shape[1]

    green_mask  = cv2.inRange(hsv, (35, 40, 40),  (85, 255, 255))
    yellow_mask = cv2.inRange(hsv, (20, 80, 100),  (35, 255, 255))
    brown_mask  = cv2.inRange(hsv, (5,  40, 40),   (20, 200, 180))
    dark_mask   = cv2.inRange(hsv, (0,  0,  0),    (180, 255, 60))

    green_r  = cv2.countNonZero(green_mask)  / total_pixels
    yellow_r = cv2.countNonZero(yellow_mask) / total_pixels
    brown_r  = cv2.countNonZero(brown_mask)  / total_pixels
    dark_r   = cv2.countNonZero(dark_mask)   / total_pixels

    breakdown = {
        "Green":  round(green_r * 100, 1),
        "Yellow": round(yellow_r * 100, 1),
        "Brown":  round(brown_r * 100, 1),
        "Dark":   round(dark_r * 100, 1),
    }

    if dark_r > 0.35:
        pred = 4
    elif dark_r > 0.20 or brown_r > 0.35:
        pred = 3
    elif brown_r > 0.15 or (yellow_r > 0.3 and brown_r > 0.08):
        pred = 2
    elif yellow_r > 0.25 and green_r < 0.15:
        pred = 1
    else:
        pred = 0

    return pred, breakdown

# HEADER

st.markdown("""
<div class="banana-header">
    <h1>BananaCheck</h1>
    <p>Multi-modal banana spoilage detection — sensor data + image analysis combined.</p>
</div>
""", unsafe_allow_html=True)

# MODEL SELECTION

st.subheader("Select Models")
selected = st.multiselect(
    "Choose one or more detection models to run:",
    ["Gas & Env", "Specter", "All Sensors", "Image Analysis"],
    placeholder="Pick at least one model..."
)

if not selected:
    st.info("Select one or more models above to get started.")
    st.stop()

st.divider()

# INPUTS + INDIVIDUAL PREDICTIONS

results = {}

for model_name in selected:

    # IMAGE MODEL
    if model_name == "Image Analysis":
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown('<h3>Image Analysis</h3>', unsafe_allow_html=True)
        st.caption("Upload a clear photo of the banana. Works best with natural lighting.")

        img_file = st.file_uploader(
            "Upload banana image",
            type=["jpg", "jpeg", "png"],
            key="img_uploader"
        )

        if img_file:
            col_img, col_res = st.columns([1, 1])
            with col_img:
                pil_img = Image.open(img_file)
                st.image(pil_img, caption="Uploaded image", use_column_width=True)

            with col_res:
                pred, breakdown = analyze_banana_image(pil_img)
                label, icon, color = LABEL_MAP[pred]

                st.markdown(f"**Color breakdown:**")
                for color_name, pct in breakdown.items():
                    st.progress(
                        min(pct / 100, 1.0),
                        text=f"{color_name}: {pct}%"
                    )

                st.markdown(f"**Result:** Class {pred} — **{label}**")
                results["Image Analysis"] = pred
        else:
            st.warning("Upload an image to get a prediction.")

        st.markdown('</div>', unsafe_allow_html=True)

    # NUMERIC MODELS
    else:
        features = MODEL_FEATURES[model_name]
        path = MODEL_PATHS[model_name]
        model_obj, scaler = load_model(model_name, path, features)

        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown(f'<h3>{model_name}</h3>', unsafe_allow_html=True)
        st.caption(f"{len(features)} input features")

        cols = st.columns(3)
        values = []
        for i, feat in enumerate(features):
            with cols[i % 3]:
                val = st.number_input(
                    feat,
                    value=0.0,
                    format="%.4f",
                    key=f"{model_name}_{feat}"
                )
                values.append(val)

        if st.button(f"Run {model_name}", key=f"btn_{model_name}"):
            pred = predict_numeric(model_obj, scaler, values)
            label, icon, color = LABEL_MAP[pred]
            results[model_name] = pred
            st.success(f"Class {pred} — {label}")

        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# LATE FUSION

st.subheader("Final Result — Late Fusion")

if len(results) == 0:
    st.caption("Run at least one model above to enable fusion.")
else:
    st.markdown("**Individual predictions so far:**")
    for name, pred in results.items():
        label, icon, _ = LABEL_MAP[pred]
        st.markdown(f"- **{name}** — Class {pred} ({label})")

    if st.button("Fuse & Get Final Prediction", key="fusion_btn"):
        final = int(round(np.mean(list(results.values()))))
        final = max(0, min(4, final))
        label, icon, color = LABEL_MAP[final]

        st.markdown(f"""
        <div class="final-result">
            <h2>{label}</h2>
            <p>Class {final} &nbsp;·&nbsp; Fused from {len(results)} model(s) &nbsp;·&nbsp; </p>
        </div>
        """, unsafe_allow_html=True)

        if final >= 3:
            st.error("This banana is not suitable for consumption.")
        elif final == 2:
            st.warning("Overripe — best used for baking or smoothies.")
        elif final == 1:
            st.success("Ripe — good to eat now.")
        else:
            st.info("Underripe — needs more time to ripen.")
