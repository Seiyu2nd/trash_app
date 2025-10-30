# ==============================
# 学習スクリプト
# 危険ゴミ分類 (battery / spray / lighter)
# ==============================

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import json

# 設定
img_size = (224, 224)
batch_size = 16
epochs = 50

# データ生成
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,         # ← 回転角をもっと大きく
    width_shift_range=0.3,     # ← 横移動を増やす
    height_shift_range=0.3,    # ← 縦移動も増やす
    shear_range=0.3,           # ← 斜め方向に歪ませる
    zoom_range=0.4,            # ← ズームの幅を広げる
    brightness_range=[0.6,1.4],# ← 明るさを変える
    horizontal_flip=True,      # ← 左右反転
    vertical_flip=True,        # ← 上下反転（使える場合）
    fill_mode='nearest',       # ← 変形でできた空白を埋める方法
    validation_split=0.2
)

# データの水増し（.flow_from_directory）
train = datagen.flow_from_directory('train', target_size=img_size,
                                    batch_size=batch_size, class_mode='categorical', subset='training')
val = datagen.flow_from_directory('train', target_size=img_size,
                                  batch_size=batch_size, class_mode='categorical', subset='validation')

# モデル構築
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3)) # include_top=Falseで分類器の部分は使わず、畳み込みの部分だけ使う                                                                                 
for layer in base.layers: layer.trainable = False                                  # →畳み込みの部分は画像の一般的・普遍的な特徴をとらえているが、分類器は以前学習したタスクに特化しているため

# 出力層を構築                                                                      # layer.trainable = Falseで凍結させる（新しく付け足した分類器の重みの更新をしている間に、学習済みモデルの重みも一緒に更新されないように凍結する）
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(train.num_classes, activation='softmax')(x)
model = Model(base.input, out)

# コンパイル（学習方法の設定）
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
# TensorFlow / Keras では、学習の途中で自動的に特定の処理を実行する機能
callbacks = [
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)

# 保存
# best_model.h5 → TensorFlowが自動で「最高の結果」のモデルを保存
# final_trash_model.h5 → 自分で手動で「最終状態」のモデルを保存
model.save('final_trash_model.h5')
print("モデル保存完了")

with open("history.json", "w") as f:
    json.dump(history.history, f)

print("学習履歴を history.json に保存しました。")
# 学習可視化

# ---- Accuracy ----
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ---- Loss ----
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# 青線（train）→ 訓練データ（train） に対する正解率。
# AIが「学習に使ったデータ」でどれだけ正しく分類できているかを示します。

# オレンジ線（val）→ 検証データ（validation） に対する正解率。
# AIが「見たことのないデータ」でどれだけ正しく分類できるかを示します。

# loss（train）と val_loss（validation）の両方が下がっていく
# そして最終的にほぼ横ばいになる
# val_loss が train_loss より極端に高くならなければ、過学習していない良い学習です