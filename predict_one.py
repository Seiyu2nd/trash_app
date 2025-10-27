# ==============================
# 推論スクリプト
# 対象: 乾電池・スプレー缶・ライター
# ==============================

import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 1. クラス名を自動取得
# 学習に使用したフォルダ構造からクラス名を再取得
dataset = image_dataset_from_directory(
    r"C:\Garbage_sorting\trash_app\train", image_size=(224, 224), shuffle=False
)
class_names = dataset.class_names
print("クラス名リスト:", class_names)

# 2. モデル読込
model_path = "final_trash_model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

model = load_model(model_path)
print("✔ モデル読み込み完了")

# 3. 推論対象の画像読込
img_path = r"C:\Garbage_sorting\trash_app\test\lighter\lighter.png"  # 判別したい画像パス

if not os.path.exists(img_path):
    raise FileNotFoundError(f"画像ファイルが見つかりません: {img_path}")

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # (224,224,3) → (1,224,224,3)
x /= 255.0  # 正規化（学習時と同じ前処理）

# 4. 予測処理
pred = model.predict(x)
predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

# 5. 結果表示
jp_labels = {"battery": "乾電池", "spray": "スプレー缶", "lighter": "ライター"}

label_jp = jp_labels.get(predicted_class, predicted_class)
print(f"分類結果: {label_jp}（{confidence*100:.2f}% の確率）")
