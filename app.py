import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
import zipfile
import pathlib

# ===== 日本語フォント設定 =====
# プロジェクト直下にフォントファイルを置く／ダウンロードして使う
font_filename = "ipaexg.ttf"
if not os.path.isfile(font_filename):
    url = "https://moji.or.jp/wp-content/ipafont/IPAexfont/IPAexfont00401.zip"
    urllib.request.urlretrieve(url, "IPAexfont.zip")
    with zipfile.ZipFile("IPAexfont.zip", "r") as z:
        z.extractall(".")
    os.rename("IPAexfont00401/ipaexg.ttf", font_filename)

font_path = pathlib.Path(font_filename).resolve()
jp_prop = fm.FontProperties(fname=str(font_path))
# フォント登録
fm.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = jp_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号が□になるのを回避

# ===== 設定 =====
TRAIN_DIR = "train"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_trash_model.h5")

# ===== タイトル =====
st.title("♻️ ごみ分類AI")
st.write("（乾電池・スプレー缶・ライター）")
st.write("カメラまたは画像ファイルから分類を行います。")

# ===== モデル・クラス読み込み =====
@st.cache_resource
def load_trash_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"モデルファイルが見つかりません: {MODEL_PATH}")
        st.stop()
    model = load_model(MODEL_PATH)
    dataset = image_dataset_from_directory(TRAIN_DIR, image_size=(224, 224), shuffle=False)
    class_names = dataset.class_names
    return model, class_names

model, class_names = load_trash_model()
jp_labels = {"battery": "乾電池", "spray": "スプレー缶", "lighter": "ライター"}

# ===== 入力方法 =====
option = st.radio("画像の取得方法を選択してください", ["📁 ファイルをアップロード", "📸 カメラで撮影"])
uploaded_image = None
if option == "📁 ファイルをアップロード":
    uploaded_image = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])
elif option == "📸 カメラで撮影":
    camera_photo = st.camera_input("カメラで撮影")
    if camera_photo is not None:
        uploaded_image = camera_photo

# ===== 推論処理 =====
if uploaded_image is not None:
    st.image(uploaded_image, caption="入力画像", use_container_width=True)
    img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    label_jp = jp_labels.get(predicted_class, predicted_class)
    st.success(f"### 判定結果: **{label_jp}**（確信度 {confidence*100:.2f}%）")

    st.subheader("📊 各クラスの予測確率")
    probs = pred[0]
    jp_class_names = [jp_labels.get(c, c) for c in class_names]

    fig, ax = plt.subplots()
    ax.barh(jp_class_names, probs, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("確率", fontproperties=jp_prop)
    ax.set_title("分類確率", fontproperties=jp_prop)
    for i, v in enumerate(probs):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontproperties=jp_prop)
    st.pyplot(fig)
else:
    st.info("📷 画像をアップロードするか、カメラで撮影してください。")
