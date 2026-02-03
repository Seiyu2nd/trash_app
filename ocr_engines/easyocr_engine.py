import easyocr
import numpy as np
from PIL import Image

# 初期化（1回だけ）
reader = easyocr.Reader(['ja', 'en'])

# 危険語キーワード
SPRAY_KEYWORDS = ["火気", "高温", "エアゾール", "ガス", "噴射"]
LIGHTER_KEYWORDS = ["ライター", "注入", "可燃性", "ガスライター"]
BATTERY_KEYWORDS = ["乾電池", "単三", "単四", "リチウム", "電池"]


def classify_by_keywords(text: str):
    if any(k in text for k in SPRAY_KEYWORDS):
        return "スプレー缶", "火気・高温などの表記が確認されたため"
    if any(k in text for k in LIGHTER_KEYWORDS):
        return "ライター", "可燃性ガス・ライター表記が確認されたため"
    if any(k in text for k in BATTERY_KEYWORDS):
        return "乾電池", "電池に関する表記が確認されたため"

    return "不明", "判定に十分な危険語句が検出されなかったため"


def run_easyocr(image: Image.Image):
    """
    ChatGPT / Cloud Vision と互換の返却形式
    """
    img = np.array(image)
    results = reader.readtext(img)

    texts = []
    confidences = []

    for _, text, conf in results:
        texts.append(text)
        confidences.append(conf)

    ocr_text = "\n".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    label, reason = classify_by_keywords(ocr_text)

    disposal_map = {
        "スプレー缶": "中身を使い切り、自治体の危険ごみ区分で廃棄してください。",
        "ライター": "ガスを抜いてから危険ごみとして廃棄してください。",
        "乾電池": "電極にテープを貼り、回収ボックスへ出してください。",
        "不明": "自治体の分別ルールを確認してください。"
    }

    return {
        "engine": "easyocr",
        "text": ocr_text,
        "confidence": round(avg_conf, 2),
        "label": label,
        "reason": reason,
        "disposal": disposal_map[label]
    }
