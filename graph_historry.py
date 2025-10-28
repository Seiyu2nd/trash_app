import json
import matplotlib.pyplot as plt

# 履歴を読み込み
with open("history.json", "r") as f:
    history = json.load(f)

# ---- Accuracy ----
plt.figure(figsize=(8, 4))
plt.plot(history['accuracy'], label='train acc')
plt.plot(history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ---- Loss ----
plt.figure(figsize=(8, 4))
plt.plot(history['loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
