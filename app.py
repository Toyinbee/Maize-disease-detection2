# app.py
import os
import io
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import tensorflow as tf

# -------------------------
# CONFIG
# -------------------------
TFLITE_MODEL_PATH = Maize_model_h5_reduced.tflite"   # <-- change to your tflite filename
IMG_SIZE = (224, 224)
RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CSV_PATH = RESULTS_DIR / "results.csv"
CONFIDENCE_THRESHOLD = 0.50  # fallback threshold (50%)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Class labels & disease info (order MUST match model output)
# -------------------------
class_labels = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy',
    'maize ear rot',
    'maize fall armyworm',
    'maize stem borer'
]

disease_info = {
    'Blight': "Blight: fungal disease causing browning/lesions. Use fungicide and avoid overhead watering.",
    'Common_Rust': "Common Rust: fungal pustules. Use resistant varieties and apply fungicide if severe.",
    'Gray_Leaf_Spot': "Gray Leaf Spot: rectangular lesions. Rotate crops and consider fungicide.",
    'Healthy': "Healthy: No obvious disease. Continue good farm practices.",
    'maize ear rot': "Maize Ear Rot: fungus on kernels. Harvest early and dry properly; avoid contaminated seed.",
    'maize fall armyworm': "Fall Armyworm: caterpillar pest. Use approved insecticides or biological controls.",
    'maize stem borer': "Stem Borer: tunnels in stems. Use resistant hybrids and biological control."
}

# -------------------------
# Load TFLite model
# -------------------------
@st.cache_resource
def load_tflite_interpreter(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_interpreter(TFLITE_MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Failed to load TFLite model: {e}")
    st.stop()

# -------------------------
# Helpers
# -------------------------
def preprocess_pil(img: Image.Image, size=IMG_SIZE):
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    # Cast to input dtype expected by the interpreter
    input_dtype = input_details[0]['dtype']
    return arr.astype(input_dtype)

def infer_tflite(interpreter, img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    probs = np.squeeze(out)
    # normalize if needed
    if probs.sum() <= 0 or probs.sum() > 1.0001:
        exp = np.exp(probs - np.max(probs))
        probs = exp / exp.sum()
    return probs

def save_result_record(image_bytes: bytes, filename: str, predicted: str, confidence: float):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{ts}__{filename}"
    image_path = IMAGES_DIR / saved_name
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "image_filename": saved_name,
        "predicted_class": predicted,
        "confidence": float(confidence)
    }
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, index=False)
    return saved_name

def annotate_image(img: Image.Image, text: str):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except Exception:
        font = ImageFont.load_default()
    w, h = img.size
    padding = 8
    text_w, text_h = draw.textsize(text, font=font)
    rect_h = text_h + 2*padding
    # semi-transparent rectangle (solid for PNG)
    draw.rectangle([(0,0),(w, rect_h)], fill=(0,0,0))
    draw.text((padding, padding), text, fill="white", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def get_history_df():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    else:
        return pd.DataFrame(columns=["timestamp_utc","image_filename","predicted_class","confidence"])

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Maize Disease Detector", page_icon="🌽", layout="centered")
st.title("🌽 Maize Disease Detection (TFLite)")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("📤 Upload maize leaf image", type=["jpg","jpeg","png"])
    threshold = st.slider("Confidence fallback threshold (%)", min_value=30, max_value=90, value=int(CONFIDENCE_THRESHOLD*100))
    CONFIDENCE_THRESHOLD = threshold / 100.0

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Running model inference..."):
                img_arr = preprocess_pil(image)
                probs = infer_tflite(interpreter, img_arr)
                pred_idx = int(np.argmax(probs))
                top_prob = float(probs[pred_idx])
                pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else "Unknown"
                is_known = (top_prob >= CONFIDENCE_THRESHOLD)
                display_label = pred_label if is_known else "Unknown / Not in model"
                confidence_pct = top_prob * 100

            # Display only the TOP prediction (big and user-friendly)
            st.markdown(
                f"""
                <div style="background:#f7f9f8; padding:18px; border-radius:10px; border:1px solid #e6efe6;">
                  <h2 style="margin:0">🔎 Prediction: <strong>{display_label}</strong></h2>
                  <p style="margin:6px 0 0 0; font-size:20px">🎯 Confidence: <strong>{confidence_pct:.2f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # progress bar for confidence
            prog = st.progress(min(100, max(0, int(confidence_pct))))
            prog.progress(int(confidence_pct))

            # Show disease advice if known, otherwise fallback
            if display_label in disease_info:
                st.info(disease_info[display_label])
            else:
                st.warning("This image's top prediction is below the confidence threshold or not recognized. Try a clearer image or different angle.")

            # OPTIONAL: allow user to view full probabilities in an expander (hidden by default)
            with st.expander("🔽 Show full class probabilities (advanced)"):
                prob_df = pd.DataFrame({
                    "class": class_labels,
                    "probability_%": (probs * 100).round(2)
                }).sort_values("probability_%", ascending=False).reset_index(drop=True)
                st.table(prob_df)

            # Save the result & offer downloads
            uploaded_bytes = uploaded.getvalue() if hasattr(uploaded, "getvalue") else None
            saved_name = save_result_record(uploaded_bytes, uploaded.name, display_label, confidence_pct)
            st.success(f"✅ Result saved as `{saved_name}`")

            # annotated image download
            annotated_text = f"{display_label} — {confidence_pct:.2f}%"
            annotated_bytes = annotate_image(image.copy(), annotated_text)
            st.download_button("⬇️ Download annotated image (PNG)", annotated_bytes, file_name=f"annotated__{saved_name}.png", mime="image/png")

            # single-row CSV download
            one_row_df = pd.DataFrame([{
                "timestamp_utc": datetime.utcnow().isoformat(),
                "image_filename": saved_name,
                "predicted_class": display_label,
                "confidence": float(confidence_pct)
            }])
            st.download_button("⬇️ Download result (CSV)", one_row_df.to_csv(index=False).encode("utf-8"), file_name=f"result__{saved_name}.csv", mime="text/csv")

with col2:
    st.subheader("📁 Saved results history")
    history_df = get_history_df()
    if history_df.empty:
        st.info("No saved results yet.")
    else:
        st.dataframe(history_df.sort_values("timestamp_utc", ascending=False).reset_index(drop=True), height=300)

        idx = st.number_input("Select result row index (0 = newest row)", min_value=0, max_value=max(0, len(history_df)-1), value=0, step=1)
        try:
            selected = history_df.sort_values("timestamp_utc", ascending=False).reset_index(drop=True).loc[idx]
            st.write("**Selected result**")
            st.write(selected.to_dict())
            img_path = IMAGES_DIR / selected["image_filename"]
            if img_path.exists():
                st.image(Image.open(img_path), caption=selected["image_filename"], use_column_width=True)
                with open(img_path, "rb") as f:
                    st.download_button("⬇️ Download original image", f.read(), file_name=selected["image_filename"], mime="image/png")
            else:
                st.warning("Saved image not found.")
        except Exception as e:
            st.error(f"Selection error: {e}")

    if not history_df.empty:
        csv_all = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download all results (CSV)", csv_all, file_name="all_results.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with Streamlit • Model: TFLite • Results saved to ./results")
