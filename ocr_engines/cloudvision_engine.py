from google.cloud import vision
from PIL import Image
import io

# =================================================
# クラス定義（単一ソース）
# =================================================
CLASSES = {
    "spray": {
        "ocr": ["火気", "高温", "可燃性", "エアゾール", "ガス", "LPG"],
        "label": ["spray", "aerosol", "can"],
        "disposal": "中身を使い切り、穴を開けずに自治体指定の方法で捨ててください。"
    },
    "lighter": {
        "ocr": ["ライター", "注入", "可燃性ガス", "火気厳禁"],
        "label": ["lighter"],
        "disposal": "ガスを完全に抜き、不燃ごみとして処分してください。"
    },
    "battery": {
        "ocr": ["乾電池", "電池", "単三", "単四", "LR6", "LR03"],
        "label": ["battery"],
        "disposal": "電極にテープを貼り、回収ボックスまたは不燃ごみとして出してください。"
    },
    "light": {
        "ocr": [],
        "label": ["light bulb", "bulb", "lamp"],
        "disposal": "割れないように紙で包み、不燃ごみとして出してください。"
    },
    "box_cutter": {
        "ocr": [],
        "label": ["box cutter", "utility knife", "cutter"],
        "disposal": "刃を紙で包み、『刃物』と表示して不燃ごみに出してください。"
    },
    "scissors": {
        "ocr": [],
        "label": ["scissors"],
        "disposal": "刃を紙で包み、『刃物』と表示して不燃ごみに出してください。"
    }
}

# =================================================
# マルチモーダル判定
# =================================================
def classify_by_multimodal(ocr_text: str, labels: list[str]):

    scores = {k: 0 for k in CLASSES}
    reasons = {k: [] for k in CLASSES}

    # OCR（重み2）
    for cls, info in CLASSES.items():
        for w in info["ocr"]:
            if w in ocr_text:
                scores[cls] += 2
                reasons[cls].append(f"OCR:{w}")

    # LABEL（重み1）
    for cls, info in CLASSES.items():
        for l in labels:
            for h in info["label"]:
                if h in l.lower():
                    scores[cls] += 1
                    reasons[cls].append(f"IMAGE:{l}")

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return (
            "unknown",
            "OCR・画像認識ともに有効な情報が検出されませんでした。",
            "自治体の案内に従って処分してください。"
        )

    return (
        best,
        " / ".join(reasons[best]),
        CLASSES[best]["disposal"]
    )

# =================================================
# メイン関数
# =================================================
def run_cloudvision_multimodal(image: Image.Image):
    try:
        client = vision.ImageAnnotatorClient()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content = buf.getvalue()

        vision_image = vision.Image(content=content)

        # OCR
        ocr_res = client.text_detection(image=vision_image)
        ocr_text = (
            ocr_res.text_annotations[0].description
            if ocr_res.text_annotations else ""
        )

        # LABEL
        label_res = client.label_detection(image=vision_image)
        labels = [l.description for l in label_res.label_annotations]

        label, reason, disposal = classify_by_multimodal(
            ocr_text,
            labels
        )

        return {
            "text": ocr_text,
            "labels": labels,
            "label": label,
            "reason": reason,
            "disposal": disposal
        }

    except Exception as e:
        return {
            "text": "",
            "labels": [],
            "label": "error",
            "reason": f"[Cloud Vision Error] {e}",
            "disposal": ""
        }
