import streamlit as st
import numpy as np
import base64
import openai
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
import zipfile
import pathlib

# ===== OpenAI APIキー設定 =====
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ===== 日本語フォント設定 =====
font_filename = "ipaexg.ttf"
if not os.path.isfile(font_filename):
    url = "https://moji.or.jp/wp-content/ipafont/IPAexfont/IPAexfont00401.zip"
    urllib.request.urlretrieve(url, "IPAexfont.zip")
    with zipfile.ZipFile("IPAexfont.zip", "r") as z:
        z.extractall(".")
    os.rename("IPAexfont00401/ipaexg.ttf", font_filename)

font_path = pathlib.Path(font_filename).resolve()
jp_prop = fm.FontProperties(fname=str(font_path))
fm.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = jp_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ===== タイトル =====
st.title("ごみ判定アプリ × GPT-4o Vision OCR")
st.write("アップロードした画像から文字をGPT-4oで読み取り、MobilenetV2で分類します。")

# ===== モデル読み込み =====
@st.cache_resource
def load_trash_model():
    model = load_model(r"C:\Garbage_sorting\trash_app\best_model.h5")
    dataset = image_dataset_from_directory("train", image_size=(224, 224), shuffle=False)
    class_names = dataset.class_names
    return model, class_names

model, class_names = load_trash_model()
jp_labels = {"battery": "乾電池", "spray": "スプレー缶", "lighter": "ライター"}

# ===== Base64変換関数 =====
def image_to_base64(uploaded_image):
    img_bytes = uploaded_image.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

# ===== GPT-4o Vision OCR関数 =====
def perform_ocr_with_gpt4o(uploaded_image):
    # MIMEタイプを自動判定
    file_type = "png"
    if hasattr(uploaded_image, "type") and uploaded_image.type:
        if "jpeg" in uploaded_image.type:
            file_type = "jpeg"
        elif "jpg" in uploaded_image.type:
            file_type = "jpeg"
        elif "webp" in uploaded_image.type:
            file_type = "webp"
        elif "gif" in uploaded_image.type:
            file_type = "gif"

    base64_image = image_to_base64(uploaded_image)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "この画像内に含まれる日本語テキストを正確にすべて抽出してください。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/{file_type};base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500,
        )
        ocr_text = response.choices[0].message.content.strip()
        return ocr_text
    except Exception as e:
        st.error(f"OCRエラー: {e}")
        return ""

# ===== 画像分類関数 =====
def classify_image(uploaded_image, model):
    img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)
    predicted_class = class_names[np.argmax(pred)]  # ← 修正済み
    confidence = np.max(pred)
    return predicted_class, confidence

# ===== 画像アップロード =====
uploaded_image = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png", "gif", "webp"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="アップロード画像", use_container_width=True)

    with st.spinner("GPT-4oで文字を認識中..."):
        ocr_result = perform_ocr_with_gpt4o(uploaded_image)

    with st.spinner("MobilenetV2でごみの種類を判定中..."):
        predicted_class, confidence = classify_image(uploaded_image, model)

    st.subheader("📄 OCR結果（GPT-4o）")
    st.text(ocr_result or "（文字が認識されませんでした）")

    # 🔥 OCR結果に特定ワードが含まれていればスプレー缶扱い
    spray_keywords = ["火気と高温に注意", "高温", "可燃", "スプレー"]
    if any(keyword in ocr_result for keyword in spray_keywords):
        final_label = "スプレー缶（文字情報から判定）"
    else:
        # 通常の画像分類結果
        label_translated = jp_labels.get(predicted_class, predicted_class)
        final_label = f"{label_translated}（AI分類）"

    st.subheader("🗑 ごみ判定結果")
    st.write(f"**分類結果：{final_label}**")
    st.write(f"**信頼度：{confidence:.2f}**")

        # 確率バー表示
    st.subheader("📊 クラス確率")
    img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)[0]

    # ← クラス名を日本語に変換
    class_labels_jp = [jp_labels.get(c, c) for c in class_names]

    fig, ax = plt.subplots()
    ax.barh(class_labels_jp, pred, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("確率", fontproperties=jp_prop)
    for i, v in enumerate(pred):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontproperties=jp_prop)
    st.pyplot(fig)
