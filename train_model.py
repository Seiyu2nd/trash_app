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

# 設定
img_size = (224, 224)
batch_size = 32
epochs = 15

# データ生成
datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
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

# 学習可視化
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend(); plt.title('Accuracy'); plt.show()


