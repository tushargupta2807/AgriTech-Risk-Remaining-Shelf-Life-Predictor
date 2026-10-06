"""
Produce Spoilage Risk & Shelf Life Predictor
----------------------------------------------
A Streamlit app that asks one question per feature (MCQ buttons wherever
possible, sliders for the two continuous sensor readings), then feeds the
answers into the two models we trained in the notebook (classification +
regression) and shows both predictions.

Run with:  streamlit run app.py

Make sure these 7 files (saved by the notebook's joblib step) are in the
SAME folder as this app.py:
    spoilage_risk_model.pkl
    shelf_life_model.pkl
    ordinal_encoder.pkl
    scaler.pkl
    ordinal_cols.pkl
    model_columns.pkl
    reverse_risk_map.pkl
"""

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG (must be the first Streamlit command)
# ==========================================================
st.set_page_config(
    page_title="Produce Spoilage Predictor",
    page_icon="🍏",
    layout="wide",
)

# ==========================================================
# CUSTOM CSS - gives the app its own look instead of default Streamlit style
# ==========================================================
st.markdown(
    """
    <style>
        /* overall page background */
        .stApp {
            background: linear-gradient(180deg, #f4faf3 0%, #ffffff 40%);
        }

        /* big header banner */
        .header-banner {
            background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 60%, #ffb74d 100%);
            padding: 28px 32px;
            border-radius: 16px;
            margin-bottom: 22px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        }
        .header-banner h1 {
            color: white;
            margin: 0;
            font-size: 32px;
        }
        .header-banner p {
            color: #eafaf1;
            margin: 6px 0 0 0;
            font-size: 15px;
        }

        /* section card wrapper */
        .section-card {
            background: white;
            border-radius: 14px;
            padding: 18px 22px 6px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #edf2ee;
        }
        .section-title {
            font-size: 19px;
            font-weight: 700;
            color: #2e7d32;
            margin-bottom: 4px;
        }
        .section-sub {
            font-size: 13px;
            color: #6b7a70;
            margin-bottom: 14px;
        }

        /* FORCE ALL FORM / MCQ TEXT TO STAY VISIBLE ON LIGHT BACKGROUND */
        .stRadio label,
        .stRadio label p,
        .stRadio label div,
        .stRadio [data-testid="stMarkdownContainer"],
        .stRadio [data-testid="stMarkdownContainer"] p,
        .stSelectbox label,
        .stSelectbox label p,
        .stSelectbox [data-testid="stMarkdownContainer"] p,
        .stCheckbox label,
        .stCheckbox label p,
        .stSlider label,
        .stSlider label p {
            color: #333333 !important;
            opacity: 1 !important;
        }

        /* radio/checkbox option text */
        div[data-baseweb="radio"] label,
        div[data-baseweb="radio"] label div,
        div[data-baseweb="radio"] label span,
        div[data-baseweb="checkbox"] label,
        div[data-baseweb="checkbox"] label div,
        div[data-baseweb="checkbox"] label span {
            color: #333333 !important;
            opacity: 1 !important;
        }

        /* progress text */
        .progress-text {
            font-size: 14px;
            color: #444;
            margin-bottom: 6px;
        }

        /* result cards */
        .result-card {
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.10);
        }
        .result-label {
            font-size: 15px;
            opacity: 0.85;
            margin-bottom: 6px;
        }
        .result-value {
            font-size: 34px;
            font-weight: 800;
        }
        .risk-low   { background: #e8f8ee; border: 2px solid #2e7d32; color: #1b5e20; }
        .risk-medium{ background: #fff8e1; border: 2px solid #f9a825; color: #8d6200; }
        .risk-high  { background: #fdecea; border: 2px solid #e53935; color: #b71c1c; }
        .shelf-card { background: #eef4fb; border: 2px solid #1e88e5; color: #0d47a1; }

        /* prediction results heading text */
        h2, h4 {
            color: #000000 !important;
        }

        /* probability bars */
        .prob-row { display:flex; align-items:center; margin-bottom:8px; }
        .prob-label { width:90px; font-size:13px; font-weight:600; color:#444; }
        .prob-track { flex:1; background:#eee; border-radius:8px; height:16px; overflow:hidden; margin:0 10px; }
        .prob-fill { height:100%; border-radius:8px; }
        .prob-pct { width:48px; font-size:13px; color:#444; text-align:right; }

        /* recommendation box */
        .reco-box {
            margin-top: 18px;
            padding: 16px 20px;
            border-radius: 12px;
            background: #f7f9f7;
            border-left: 5px solid #2e7d32;
            font-size: 15px;
            color: #333;
        }

        /* missing-answer warning list */
        .missing-box {
            background: #fff3f2;
            border: 1px solid #f5c2c0;
            border-radius: 10px;
            padding: 14px 18px;
            color: #8a1f1a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# LOAD SAVED MODEL FILES (cached so they load only once)
# ==========================================================
@st.cache_resource
def load_artifacts():
    clf = joblib.load("spoilage_risk_model.pkl")
    reg = joblib.load("shelf_life_model.pkl")
    ord_enc = joblib.load("ordinal_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    ordinal_cols = joblib.load("ordinal_cols.pkl")
    model_columns = joblib.load("model_columns.pkl")
    reverse_risk_map = joblib.load("reverse_risk_map.pkl")
    return clf, reg, ord_enc, scaler, ordinal_cols, model_columns, reverse_risk_map


clf_model, reg_model, ord_enc, scaler, ORDINAL_COLS, MODEL_COLUMNS, REVERSE_RISK_MAP = load_artifacts()

# ==========================================================
# OPTION LISTS - these must exactly match the categories used while
# training (same spelling, same order) or the encoder will error out
# ==========================================================
PRODUCE_TYPES = ["Apple", "Banana", "Carrot", "Lettuce", "Mango",
                  "Onion", "Peach", "Potato", "Spinach", "Tomato"]

REGIONS = ["Central", "East", "North", "South", "West"]

ORDINAL_OPTIONS = {
    "ambient_temperature_level": ["Cool (<10°C)", "Moderate (10-20°C)", "Warm (20-30°C)", "Hot (>30°C)"],
    "storage_duration_bucket": ["Less than 1 day", "1-3 days", "3-7 days", "More than 7 days"],
    "transport_duration_bucket": ["Less than 6 hrs", "6-12 hrs", "12-24 hrs", "More than 24 hrs"],
    "co2_level_bucket": ["Normal", "Elevated", "High", "Very High"],
    "ethylene_level_bucket": ["Low", "Moderate", "High", "Very High"],
    "vibration_level": ["Minimal", "Mild", "Moderate", "Severe"],
    "light_exposure_level": ["Dark", "Low", "Moderate", "Bright"],
    "door_open_frequency": ["0-2 times", "3-5 times", "6-10 times", "More than 10 times"],
    "power_outage_bucket": ["No Outage", "Less than 30 min", "30-60 min", "More than 60 min"],
    "initial_quality_grade": ["Excellent", "Good", "Fair", "Poor"],
}

# question text + short helper shown above each MCQ
ORDINAL_QUESTIONS = {
    "ambient_temperature_level": "What is the ambient (surrounding air) temperature level?",
    "storage_duration_bucket": "How long has this batch been sitting in storage?",
    "transport_duration_bucket": "How long did transport from farm to warehouse take?",
    "co2_level_bucket": "What is the CO2 level around the produce?",
    "ethylene_level_bucket": "What is the ethylene gas level around the produce?",
    "vibration_level": "How much vibration/shaking did the produce experience during transport?",
    "light_exposure_level": "How much light was the produce exposed to?",
    "door_open_frequency": "How many times was the storage door opened recently?",
    "power_outage_bucket": "Was there any power outage affecting cold storage?",
    "initial_quality_grade": "What was the initial quality grade when the batch arrived?",
}

# grouping the 10 MCQs into 3 logical sections for a cleaner layout
SECTION_2_FIELDS = ["ambient_temperature_level", "storage_duration_bucket", "transport_duration_bucket"]
SECTION_3_FIELDS = ["co2_level_bucket", "ethylene_level_bucket", "vibration_level", "light_exposure_level"]
SECTION_4_FIELDS = ["door_open_frequency", "power_outage_bucket", "initial_quality_grade"]

TOTAL_QUESTIONS = 14  # produce_type + region + 2 sliders + 10 MCQs

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <div class="header-banner">
        <h1>🍏 Produce Spoilage Risk & Shelf Life Predictor</h1>
        <p>Answer every question below exactly as the sensor/warehouse data shows it.
        Once all 14 questions are answered, click "Check Score" to get the spoilage risk
        and the estimated remaining shelf life.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# SECTION 1: Produce & Location  (2 MCQ dropdowns)
# ==========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🍎 Produce & Location</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Tell us what the batch is and where it is.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    produce_type = st.selectbox(
        "Which produce type is this?",
        PRODUCE_TYPES,
        index=None,
        placeholder="Select produce type",
        key="produce_type",
    )
with col2:
    region = st.selectbox(
        "Which region is this batch located in?",
        REGIONS,
        index=None,
        placeholder="Select region",
        key="region",
    )
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# SECTION 2: Live sensor readings (the 2 SLIDERS)
# ==========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌡️ Live Sensor Readings</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Drag each slider to match the actual sensor reading, '
    'then tick the confirm box (this makes sure a value is never submitted by accident).</div>',
    unsafe_allow_html=True,
)

sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("**Storage Temperature (°C)**")
    temp_value = st.slider(
        "Storage Temperature (°C)", -10.0, 25.0, 7.5, 0.5,
        key="temp_slider", label_visibility="collapsed",
    )
    temp_confirmed = st.checkbox("✅ I've set the actual temperature reading", key="temp_confirmed")

with sc2:
    st.markdown("**Humidity (%)**")
    humidity_value = st.slider(
        "Humidity (%)", 40.0, 100.0, 70.0, 0.5,
        key="humidity_slider", label_visibility="collapsed",
    )
    humidity_confirmed = st.checkbox("✅ I've set the actual humidity reading", key="humidity_confirmed")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# SECTION 3: Storage & Transport Conditions (MCQ)
# ==========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📦 Storage & Transport Conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Pick the option that matches the batch.</div>', unsafe_allow_html=True)

mcq_answers = {}
cols = st.columns(3)
for i, field in enumerate(SECTION_2_FIELDS):
    with cols[i]:
        mcq_answers[field] = st.radio(
            ORDINAL_QUESTIONS[field],
            ORDINAL_OPTIONS[field],
            index=None,
            key=field,
        )
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# SECTION 4: Atmospheric Conditions (MCQ)
# ==========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌬️ Atmospheric Conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Pick the option that matches the batch.</div>', unsafe_allow_html=True)

cols2 = st.columns(4)
for i, field in enumerate(SECTION_3_FIELDS):
    with cols2[i]:
        mcq_answers[field] = st.radio(
            ORDINAL_QUESTIONS[field],
            ORDINAL_OPTIONS[field],
            index=None,
            key=field,
        )
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# SECTION 5: Handling & Quality (MCQ)
# ==========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔌 Handling & Quality</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Pick the option that matches the batch.</div>', unsafe_allow_html=True)

cols3 = st.columns(3)
for i, field in enumerate(SECTION_4_FIELDS):
    with cols3[i]:
        mcq_answers[field] = st.radio(
            ORDINAL_QUESTIONS[field],
            ORDINAL_OPTIONS[field],
            index=None,
            key=field,
        )
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# LIVE PROGRESS INDICATOR (updates as the user answers questions)
# ==========================================================
answered_count = sum([
    produce_type is not None,
    region is not None,
    temp_confirmed,
    humidity_confirmed,
] + [mcq_answers[f] is not None for f in ORDINAL_OPTIONS])

st.markdown(
    f'<div class="progress-text">Answered {answered_count} / {TOTAL_QUESTIONS} questions</div>',
    unsafe_allow_html=True,
)
st.progress(answered_count / TOTAL_QUESTIONS)

# ==========================================================
# ACTION BUTTONS
# ==========================================================
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    check_clicked = st.button("🔍 Check Score", use_container_width=True, type="primary")
with btn_col2:
    reset_clicked = st.button("↺ Reset", use_container_width=True)

if reset_clicked:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ==========================================================
# ON SUBMIT: validate, then predict
# ==========================================================
if check_clicked:

    # ---- 1. Build a list of anything still unanswered ----
    missing = []
    if produce_type is None:
        missing.append("Produce type")
    if region is None:
        missing.append("Region")
    if not temp_confirmed:
        missing.append("Storage temperature (tick the confirm box)")
    if not humidity_confirmed:
        missing.append("Humidity (tick the confirm box)")
    for field in ORDINAL_OPTIONS:
        if mcq_answers[field] is None:
            missing.append(ORDINAL_QUESTIONS[field])

    if missing:
        items = "".join(f"<li>{m}</li>" for m in missing)
        st.markdown(
            f'<div class="missing-box"><b>⚠️ Please answer these before checking score:</b>'
            f'<ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )

    else:
        # ---- 2. Ordinal-encode the 10 MCQ answers (same order the encoder was trained on) ----
        ordinal_input_df = pd.DataFrame([[mcq_answers[c] for c in ORDINAL_COLS]], columns=ORDINAL_COLS)
        encoded_ordinal = ord_enc.transform(ordinal_input_df)[0]
        encoded_lookup = dict(zip(ORDINAL_COLS, encoded_ordinal))

        # ---- 3. Recreate the same engineered feature used in training (Step 8) ----
        environmental_stress_score = (
            encoded_lookup["ambient_temperature_level"]
            + encoded_lookup["storage_duration_bucket"]
            + encoded_lookup["co2_level_bucket"]
            + encoded_lookup["ethylene_level_bucket"]
        )

        # ---- 4. Build one row with every column the model expects, in the exact order ----
        row = {col: 0 for col in MODEL_COLUMNS}
        row["storage_temperature_C"] = temp_value
        row["humidity_pct"] = humidity_value
        for col in ORDINAL_COLS:
            row[col] = encoded_lookup[col]
        row[f"produce_type_{produce_type}"] = 1
        row[f"region_{region}"] = 1
        row["environmental_stress_score"] = environmental_stress_score

        input_vector_df = pd.DataFrame([row], columns=MODEL_COLUMNS)
        input_scaled = scaler.transform(input_vector_df)

        # ---- 5. Predict with both models ----
        risk_num = clf_model.predict(input_scaled)[0]
        risk_label = REVERSE_RISK_MAP[risk_num]
        risk_probs = clf_model.predict_proba(input_scaled)[0]  # [P(Low), P(Medium), P(High)]

        shelf_life_hours = reg_model.predict(input_scaled)[0]
        shelf_life_days = shelf_life_hours / 24

        # ---- 6. Display results ----
        st.markdown("## 📊 Prediction Results")

        risk_css_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[risk_label]
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk_label]

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(
                f"""
                <div class="result-card {risk_css_class}">
                    <div class="result-label">Spoilage Risk</div>
                    <div class="result-value">{risk_emoji} {risk_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.markdown(
                f"""
                <div class="result-card shelf-card">
                    <div class="result-label">Estimated Remaining Shelf Life</div>
                    <div class="result-value">{shelf_life_hours:,.0f} hrs</div>
                    <div class="result-label">(~ {shelf_life_days:,.1f} days)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- probability breakdown so the user can see model confidence ----
        st.markdown("#### Model Confidence")
        prob_colors = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#e53935"}
        class_order = ["Low", "Medium", "High"]
        for cls_name, prob in zip(class_order, risk_probs):
            pct = prob * 100
            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-label">{cls_name}</div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{pct:.1f}%; background:{prob_colors[cls_name]};"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- simple recommendation text based on the predicted risk class ----
        recommendations = {
            "Low": "This batch looks safe. Continue normal storage and re-check periodically.",
            "Medium": "Keep an eye on this batch. Consider moving it up in the selling queue or "
                      "improving storage conditions (temperature/humidity) if possible.",
            "High": "Act soon — consider a price discount to sell quickly, prioritize shipping "
                    "this batch first, or move it to a better-controlled cold storage unit.",
        }
        st.markdown(
            f'<div class="reco-box"><b>💡 Suggested Action:</b> {recommendations[risk_label]}</div>',
            unsafe_allow_html=True,
        )