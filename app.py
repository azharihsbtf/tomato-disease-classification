import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_DIR = Path(__file__).parent / "models" / "v1"
MODEL_PATH = MODEL_DIR / "tomato_disease_model.tflite"
META_PATH = MODEL_DIR / "metadata.json"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["healthy", "late_blight", "septoria_leaf_spot"]
LABEL_ID = {
    "healthy": "Healthy",
    "late_blight": "Late Blight",
    "septoria_leaf_spot": "Septoria Leaf Spot",
}
CONF_THRESHOLD = 0.90


@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    arr = tf.image.resize(arr, IMG_SIZE).numpy().astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(interpreter, batch: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], batch)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]


st.set_page_config(page_title="Deteksi Penyakit Daun Tomat")
st.title("Deteksi Penyakit Daun Tomat")
st.caption("Unggah foto satu daun tomat. Model mengklasifikasikannya ke "
           "Healthy, Late Blight, atau Septoria Leaf Spot.")

with st.sidebar:
    st.header("Informasi Model")
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        st.write(f"Versi: **{meta['version']}**")
        st.write(f"Arsitektur: {meta['architecture']}")
        st.write(f"Akurasi uji: {meta['test_accuracy']:.2%}")
        st.write(f"Macro F1: {meta['macro_f1']:.3f}")
    st.info("Prototipe ini hanya alat bantu awal, bukan pengganti ahli "
            "pertanian.")

file = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    st.image(img, caption="Gambar yang diunggah", width=320)
    probs = predict(load_interpreter(), preprocess(img))
    idx = int(np.argmax(probs))
    name, conf = CLASS_NAMES[idx], float(probs[idx])

    if conf < CONF_THRESHOLD:
        st.warning(f"Keyakinan model rendah ({conf:.1%}). Coba foto yang "
                   "lebih jelas, satu daun, dengan pencahayaan cukup.")
    st.subheader(f"Prediksi: {LABEL_ID[name]}")
    st.write(f"Tingkat keyakinan: **{conf:.1%}**")
