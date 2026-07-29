import os
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import pandas as pd

st.set_page_config(page_title="Multimodal Parkinson Detection", layout="wide")

st.markdown("""
<style>
.caption-text { color: black !important; font-size: 14px; }
.metric-box {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
}
.conflict-box {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 8px 0;
}
.high-conf   { color: #1a7f37; font-weight: bold; }
.medium-conf { color: #9a6700; font-weight: bold; }
.low-conf    { color: #cf222e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Multimodal Parkinson Disease Detection System")
st.caption("Decision-level fusion of Gait · Voice · Handwriting signals with conflict-aware confidence reporting")

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
#from temporal_attention import TemporalAttention
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(inputs * a, axis=1)

@st.cache_resource
def load_models():
    gait_model = tf.keras.models.load_model(
        "PD_Project_gait_models/gait_model_FIXED.h5",
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False
    )
    hand_spiral_model = tf.keras.models.load_model(
        "PD_Project_handwriting_models/densenet_spiral_final.keras"
    )
    hand_wave_model = tf.keras.models.load_model(
        "PD_Project_handwriting_models/densenet_wave_final.keras"
    )
    voice_ensemble = joblib.load("PD_voice_models/ensemble_model.pkl")
    voice_scaler   = joblib.load("PD_voice_models/scaler.pkl")
    voice_pca      = joblib.load("PD_voice_models/pca.pkl")
    return gait_model, hand_spiral_model, hand_wave_model, voice_ensemble, voice_scaler, voice_pca

gait_model, hand_spiral_model, hand_wave_model, voice_ensemble, voice_scaler, voice_pca = load_models()

# ─────────────────────────────────────────────────────────────
# GRID-SEARCH OPTIMISED WEIGHTS  (from fusion_revised.py)
# w_gait=0.20  w_voice=0.30  w_hand=0.50
# Bootstrap accuracy: 81.6% ± 3.73%   AUC: 0.8952
# ─────────────────────────────────────────────────────────────
BASE_WEIGHTS = {"gait": 0.20, "voice": 0.30, "hand": 0.50}

GAIT_FEATURES = [
    "Time","L1","L2","L3","L4","L5","L6","L7","L8",
    "R1","R2","R3","R4","R5","R6","R7","R8",
    "Total_Force_Left","Total_Force_Right"
]

VOICE_FEATURE_NAMES = [
    "MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)",
    "MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP",
    "MDVP:PPQ","Jitter:DDP","MDVP:Shimmer",
    "MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5",
    "MDVP:APQ","Shimmer:DDA","NHR","HNR",
    "RPDE","DFA","spread1","spread2","D2","PPE"
]

# ─────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────
def preprocess_gait_txt(txt_path):
    df = pd.read_csv(txt_path, sep="\t", header=None, names=GAIT_FEATURES)
    df = df.iloc[:400]
    return df.values.T

def preprocess_handwriting_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_voice(csv_path):
    df = pd.read_csv(csv_path)
    X  = df.values
    X  = voice_scaler.transform(X)
    X  = voice_pca.transform(X)
    return X

# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict_gait(txt_path):
    g = preprocess_gait_txt(txt_path).reshape(1, 19, 400, 1)
    return float(gait_model.predict(g)[0][1])

def predict_handwriting(spiral_path, wave_path):
    sp = hand_spiral_model.predict(preprocess_handwriting_image(spiral_path))[0][0]
    wv = hand_wave_model.predict(preprocess_handwriting_image(wave_path))[0][0]
    return float(0.7 * sp + 0.3 * wv)

def predict_voice(csv_path):
    X = preprocess_voice(csv_path)
    return float(voice_ensemble.predict_proba(X)[0][1])

# ─────────────────────────────────────────────────────────────
# FUSION  (grid-search weights + conflict detection)
# ─────────────────────────────────────────────────────────────
CONFLICT_THRESHOLD = 0.15   # std of modality probs above this → conflict
DECISION_THRESHOLD = 0.50

def adaptive_fusion(gait=None, handwriting=None, voice=None):
    """
    Grid-search optimised decision-level fusion.
    Returns fused_prob, decision, confidence, conflict_flag, explanation_dict.
    """
    available = {}
    if gait        is not None: available["gait"]  = gait
    if handwriting is not None: available["hand"]  = handwriting
    if voice       is not None: available["voice"] = voice

    if not available:
        raise ValueError("Provide at least one modality.")

    # Renormalise weights to available modalities
    total_w = sum(BASE_WEIGHTS[m] for m in available)
    norm_w  = {m: BASE_WEIGHTS[m] / total_w for m in available}

    fused_prob = sum(norm_w[m] * available[m] for m in available)

    # Conflict detection
    probs_list = list(available.values())
    conflict   = (np.std(probs_list) > CONFLICT_THRESHOLD) if len(probs_list) > 1 else False

    # Confidence
    dist = abs(fused_prob - DECISION_THRESHOLD)
    if dist >= 0.30:
        confidence = "High"
    elif dist >= 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"

    if conflict and confidence == "Low":
        confidence = "Low (clinical review recommended)"

    decision = "Parkinson" if fused_prob >= DECISION_THRESHOLD else "Healthy"

    explanation = {
        m: {
            "prob":         round(available[m], 3),
            "weight":       round(norm_w[m], 3),
            "contribution": round(norm_w[m] * available[m], 3)
        }
        for m in available
    }

    return fused_prob, decision, confidence, conflict, explanation

# ─────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────
def visualize_gait_signal(gait_sample):
    mean_signal = np.mean(gait_sample, axis=0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(mean_signal, linewidth=1, color="#2563EB")
    ax.set_title("Gait Temporal Force Pattern", fontsize=9)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Average Force", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=False)

def visualize_handwriting_gradcam(model, image_path, layer_name, title):
    img = preprocess_handwriting_image(image_path)
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img)
        loss = preds[0]
    grads       = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out    = conv_out[0]
    heatmap     = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = np.maximum(heatmap, 0)
    heatmap    /= np.max(heatmap) + 1e-8
    img_orig    = cv2.imread(image_path)
    h, w, _    = img_orig.shape
    heatmap     = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap     = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay     = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=title, width=350)

def visualize_voice_pca(ensemble_model, pca_model):
    rf           = ensemble_model.named_estimators_["rf"]
    importance   = rf.feature_importances_
    components   = pca_model.components_
    labels       = []
    for i in range(len(importance)):
        max_idx = np.argmax(np.abs(components[i]))
        labels.append(VOICE_FEATURE_NAMES[max_idx])
    top_idx       = np.argsort(importance)[::-1][:10]
    top_imp       = importance[top_idx]
    top_labels    = [labels[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(top_labels, top_imp, color="#7C3AED")
    ax.invert_yaxis()
    ax.set_title("Top Voice Feature Contribution via PCA", fontsize=9)
    ax.set_xlabel("Importance", fontsize=8)
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

def visualize_probability_bars(probs_dict, fused_prob):
    labels = list(probs_dict.keys()) + ["Fusion"]
    values = list(probs_dict.values()) + [fused_prob]
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"][:len(labels)]
    colors[-1] = "#EF4444"
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("PD Probability", fontsize=8)
    ax.set_title("Per-Modality vs Fused Probability", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(val + 0.03, 1.05),
                f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

def visualize_contribution_pie(explanation):
    labels = [m.capitalize() for m in explanation]
    sizes  = [explanation[m]["contribution"] for m in explanation]
    colors = ["#3B82F6", "#10B981", "#F59E0B"][:len(labels)]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors,
           startangle=140, textprops={"fontsize": 8})
    ax.set_title("Modality Contribution to Fusion", fontsize=9)
    st.pyplot(fig, use_container_width=False)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Upload Inputs")
st.sidebar.caption("Upload any combination — system adapts automatically.")
gait_file   = st.sidebar.file_uploader("Gait data (.txt)",          type=["txt"])
spiral_file = st.sidebar.file_uploader("Spiral drawing (.png/.jpg)", type=["png","jpg"])
wave_file   = st.sidebar.file_uploader("Wave drawing (.png/.jpg)",   type=["png","jpg"])
voice_file  = st.sidebar.file_uploader("Voice features (.csv)",      type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info (grid-search optimised)**")
st.sidebar.markdown(f"- w_gait = {BASE_WEIGHTS['gait']}")
st.sidebar.markdown(f"- w_voice = {BASE_WEIGHTS['voice']}")
st.sidebar.markdown(f"- w_hand = {BASE_WEIGHTS['hand']}")
st.sidebar.markdown("Bootstrap accuracy: **81.6% ± 3.73%**")
st.sidebar.markdown("AUC-ROC: **0.8952**")

# ─────────────────────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────────────────────
if st.button("Run Prediction", type="primary"):
    gait_prob = handwriting_prob = voice_prob = None

    with st.spinner("Running models..."):
        if gait_file:
            with open("temp_gait.txt", "wb") as f: f.write(gait_file.read())
            gait_prob = predict_gait("temp_gait.txt")

        if spiral_file and wave_file:
            with open("temp_spiral.png", "wb") as f: f.write(spiral_file.read())
            with open("temp_wave.png",   "wb") as f: f.write(wave_file.read())
            handwriting_prob = predict_handwriting("temp_spiral.png", "temp_wave.png")
        elif spiral_file or wave_file:
            st.warning("Please upload BOTH spiral and wave images for handwriting analysis.")

        if voice_file:
            with open("temp_voice.csv", "wb") as f: f.write(voice_file.read())
            voice_prob = predict_voice("temp_voice.csv")

    if all(p is None for p in [gait_prob, handwriting_prob, voice_prob]):
        st.error("Please upload at least one modality input.")
        st.stop()

    fused_prob, decision, confidence, conflict, explanation = adaptive_fusion(
        gait=gait_prob, handwriting=handwriting_prob, voice=voice_prob
    )

    # ── RESULT BANNER ──
    st.markdown("---")
    if decision == "Parkinson":
        st.error(f"### 🔴 Prediction: {decision}  |  Probability: {fused_prob:.3f}")
    else:
        st.success(f"### 🟢 Prediction: {decision}  |  Probability: {fused_prob:.3f}")

    # ── CONFIDENCE & CONFLICT ──
    conf_color = "high-conf" if "High" in confidence else ("medium-conf" if "Medium" in confidence else "low-conf")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Confidence:** <span class='{conf_color}'>{confidence}</span>", unsafe_allow_html=True)
    with col_b:
        if conflict:
            st.markdown("⚠️ **Modality conflict detected** — modalities disagree significantly.", unsafe_allow_html=True)
        else:
            st.markdown("✅ Modalities are in agreement.", unsafe_allow_html=True)

    if conflict:
        st.markdown("""
        <div class='conflict-box'>
        <b>Conflict Notice:</b> The standard deviation of modality probabilities exceeds 0.15,
        indicating substantial disagreement across signals. The fused decision is still produced
        using weighted averaging, but clinical verification is recommended.
        </div>
        """, unsafe_allow_html=True)

    # ── PROBABILITY CHARTS ──
    st.markdown("---")
    st.subheader("Probability Overview")
    probs_dict = {}
    if gait_prob        is not None: probs_dict["Gait"]        = gait_prob
    if handwriting_prob is not None: probs_dict["Handwriting"] = handwriting_prob
    if voice_prob       is not None: probs_dict["Voice"]       = voice_prob

    chart_col, pie_col = st.columns(2)
    with chart_col:
        visualize_probability_bars(probs_dict, fused_prob)
    with pie_col:
        visualize_contribution_pie(explanation)

    # ── EXPLANATION TABLE ──
    st.subheader("Fusion Explanation")
    exp_rows = []
    for m, v in explanation.items():
        exp_rows.append({
            "Modality"    : m.capitalize(),
            "PD Prob"     : v["prob"],
            "Weight"      : v["weight"],
            "Contribution": v["contribution"]
        })
    st.dataframe(pd.DataFrame(exp_rows).set_index("Modality"), use_container_width=True)

    # ── XAI SECTION ──
    st.markdown("---")
    st.subheader("Explainable AI")

    if gait_prob is not None:
        with st.expander("Gait Analysis", expanded=True):
            col1, col2 = st.columns([3, 3])
            with col1:
                visualize_gait_signal(preprocess_gait_txt("temp_gait.txt"))
            with col2:
                st.markdown("""
                <p class="caption-text">
                This plot shows average vertical force variation over 400 time steps.
                Irregular magnitude and rhythm patterns are learned by the CNN-BiLSTM model
                to detect gait abnormalities associated with Parkinson disease.
                </p>""", unsafe_allow_html=True)

    if handwriting_prob is not None:
        with st.expander("Handwriting Analysis (Grad-CAM)", expanded=True):
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                visualize_handwriting_gradcam(
                    hand_spiral_model, "temp_spiral.png",
                    "conv5_block16_concat", "Spiral Grad-CAM"
                )
            with h_col2:
                visualize_handwriting_gradcam(
                    hand_wave_model, "temp_wave.png",
                    "conv5_block16_concat", "Wave Grad-CAM"
                )
            st.markdown("""
            <p class="caption-text">
            Grad-CAM highlights image regions that most influenced the DenseNet121 classification.
            Warmer colours indicate higher attention — typically around irregular curvature,
            tremor artefacts, and uneven stroke pressure.
            </p>""", unsafe_allow_html=True)

    if voice_prob is not None:
        with st.expander("Voice Analysis", expanded=True):
            visualize_voice_pca(voice_ensemble, voice_pca)
            st.markdown("""
            <p class="caption-text">
            Feature importance mapped back through PCA loadings to original acoustic features.
            Dominant features such as jitter, shimmer, and HNR reflect vocal impairments
            characteristic of Parkinson disease.
            </p>""", unsafe_allow_html=True)
