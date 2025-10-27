import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 日本語フォント設定 ---
plt.rcParams['font.family'] = 'Meiryo'

# ==============================
# 設定
# ==============================
TRAIN_DIR = r"C:\Garbage_sorting\trash_app\train"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_trash_model.h5") # r"C:\Garbage_sorting\trash_app\final_trash_model.h5"

# ==============================
# アプリタイトル
# ==============================
st.title("♻️ ごみ分類AI")
st.title("（乾電池・スプレー缶・ライター）")
st.write("カメラまたは画像ファイルから分類を行います。")

# ==============================
# モデル・クラス名の読み込み
# ==============================
@st.cache_resource # アプリ起動時に一度だけモデルとクラス名を読み込み、以降はキャッシュを使う（高速化）。
def load_trash_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"モデルファイルが見つかりません: {MODEL_PATH}")
        st.stop()
    model = load_model(MODEL_PATH)
    dataset = image_dataset_from_directory(TRAIN_DIR, image_size=(224, 224), shuffle=False) # クラスの名前を自動取得
    class_names = dataset.class_names
    return model, class_names

model, class_names = load_trash_model()
jp_labels = {"battery": "乾電池", "spray": "スプレー缶", "lighter": "ライター"}

# ==============================
# 入力方法選択
# ==============================
option = st.radio("画像の取得方法を選択してください", ["📁 ファイルをアップロード", "📸 カメラで撮影"])

uploaded_image = None

if option == "📁 ファイルをアップロード":
    uploaded_image = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])
elif option == "📸 カメラで撮影":
    camera_photo = st.camera_input("カメラで撮影")
    if camera_photo is not None:
        uploaded_image = camera_photo

# ==============================
# 推論処理
# ==============================
if uploaded_image is not None:
    st.image(uploaded_image, caption="入力画像")

    # 画像前処理
    img = Image.open(uploaded_image)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    # 推論
    pred = model.predict(x) # predにはNumPy 配列で、各クラスに対する確率のリストが返される
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) # 予測確率の最大値を取得
    label_jp = jp_labels.get(predicted_class, predicted_class) # 第一引数：検索キー（predicted_class）第二引数：該当キーがない場合のデフォルト値

    # ==============================
    # 結果表示
    # ==============================
    st.success(f"### 判定結果: **{label_jp}**（確信度 {confidence*100:.2f}%）")

    # ==============================
    # 確率棒グラフ表示
    # ==============================
    st.subheader(" 各クラスの予測確率")

    # 棒グラフ用データ
    probs = pred[0]
    jp_class_names = [jp_labels.get(c, c) for c in class_names]

    fig, ax = plt.subplots() # Matplotlib でグラフ描画用の「図 (fig)」と「軸 (ax)」を作成
    ax.barh(jp_class_names, probs, color='skyblue') # 横向きグラフ,jp_class_names が縦軸、probs が横軸
    ax.set_xlim(0, 1) # 表示範囲の設定0〜1 に固定（100%に対応）
    ax.set_xlabel("確率") # 横軸ラベルに「確率」の設定
    ax.set_title("分類確率") # グラフタイトルを「分類確率」に設定
    for i, v in enumerate(probs):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center') # 角棒の右側に確率（％）を付与、ax.text(x座標, y座標, 表示する文字列, オプション)
    st.pyplot(fig)                                         # v + 0.02 … + 0.02 をして、棒の右端から少し右にずらす // enumerate(probs) により、i は0, 1, 2, …とクラスごとに増加
                                                           # f"{v*100:.1f}%" … 表示するテキスト内容を「確率（％）」に変換 // va='center' … 垂直方向の揃え方を指定。'center' は、「y座標（=棒の高さの中央）に文字を中央揃えで配置」するという意味
                                                           # Streamlit にグラフを埋め込み表示

else:
    st.info("📷 画像をアップロードするか、カメラで撮影してください。")
