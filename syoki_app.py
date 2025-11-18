import streamlit as st
import numpy as np
from my_translation import translations
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

# ===== 言語選択UI =====
# 利用可能な言語一覧（キー）
available_languages = {
    '日本語': 'ja',
    'English': 'en'
}

# サイドバーで言語を選ぶ
lang_display = st.sidebar.selectbox("言語を選択 / Choose Language", list(available_languages.keys()))
lang_code = available_languages[lang_display]

# 選択した言語の翻訳辞書
t = translations[lang_code]

# ===== 設定 =====
TRAIN_DIR = "train"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_trash_model.h5")

# ===== タイトル =====
st.title(t['title'])
st.write(t['type'])
st.write(t['description'])

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
option = st.radio(t['input_method'], [t['fail'], t['camera']])
uploaded_image = None
if option == t['fail']:
    uploaded_image = st.file_uploader(t['fail_input'], type=["jpg", "jpeg", "png"])
elif option == t['camera']:
    camera_photo = st.camera_input(t['camera_input'])
    if camera_photo is not None:
        uploaded_image = camera_photo

# ===== 推論処理 =====
if uploaded_image is not None:
    st.image(uploaded_image, caption=t['input_image'], use_container_width=True)
    img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    # 言語ごとのラベル対応
    labels = {
        "ja": {"battery": "乾電池", "spray": "スプレー缶", "lighter": "ライター"},
        "en": {"battery": "Battery", "spray": "Spray Can", "lighter": "Lighter"},
    }
    label_translated = labels[lang_code].get(predicted_class, predicted_class)

    # 結果表示（多言語対応）
    st.success(f"{t['result']}: **{label_translated}**（{t['confidence']} {confidence*100:.2f}%）")

    # グラフ表示
    st.subheader(t['prob_chart_subtitle'])
    probs = pred[0]
    class_labels = [labels[lang_code].get(c, c) for c in class_names]

    fig, ax = plt.subplots()
    ax.barh(class_labels, probs, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel(t['prob_x_label'], fontproperties=jp_prop)
    ax.set_title(t['prob_chart_title'], fontproperties=jp_prop)
    for i, v in enumerate(probs):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontproperties=jp_prop)
    st.pyplot(fig)
else:
    st.info(t['info'])