# ocr_engines/tesseract_engine.py

import pytesseract
from PIL import Image
import cv2
import numpy as np
import streamlit as st

# =================================================
# Tesseract パス（Windows）
# =================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =================================================
# 共通OCR前処理（ライター・乾電池向け）
# =================================================
def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)

    return bin_img


# =================================================
# スプレー缶専用OCR前処理
# =================================================
def preprocess_spray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # グレースケール化

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bin_img = cv2.adaptiveThreshold( # 2値化（白黒（0 or 255））
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return bin_img


# =================================================
# メイン関数（app.py 互換・完全版）
# =================================================
def run_tesseract(image_pil):
    """
    補助OCR（Tesseract）
    ・ライター
    ・スプレー缶
    ・乾電池
    を複数手法＋可視化＋高速化で判定
    """
    """
    戻り値:
    {
        label: str,
        reason: str,
    }
    """

    # -------------------------------------------------
    # 初期化（session_state）
    # -------------------------------------------------
    st.session_state["tesseract_debug"] = {
        "images": [],
        "ocr_logs": [],
        "keyword_hits": [],
        "scores": {}
    }

    # -------------------------------------------------
    # PIL → OpenCV
    # -------------------------------------------------
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    st.session_state["tesseract_debug"]["images"].append(
        ("入力画像", img_cv)
    )

    # -------------------------------------------------
    # 赤帯検出（スプレー缶判定補助）
    # -------------------------------------------------
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 60, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 60, 40])
    upper_red2 = np.array([179, 255, 255])

    mask_red = (
        cv2.inRange(hsv, lower_red1, upper_red1) |
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    red_ratio = np.count_nonzero(mask_red) / mask_red.size

    roi_candidates = [("full", img_cv)]

    if red_ratio > 0.01:
        roi_red = cv2.bitwise_and(img_cv, img_cv, mask=mask_red)
        roi_candidates.append(("red_band", roi_red))

        st.session_state["tesseract_debug"]["images"].append(
            ("赤帯検出領域", roi_red)
        )

    # -------------------------------------------------
    # OCR（条件付き・高速化）
    # -------------------------------------------------
    ocr_logs = []
    config_fast = r"--oem 1 --psm 6 -l jpn"

    for roi_name, roi in roi_candidates:

        # --- 通常前処理 ---
        bin_normal = preprocess_for_ocr(roi)

        # 通常
        text = pytesseract.image_to_string(bin_normal, config=config_fast)
        ocr_logs.append({
            "roi": roi_name,
            "method": "normal",
            "text": text
        })

        # ★ 通常の反転
        bin_normal_inv = cv2.bitwise_not(bin_normal)
        text = pytesseract.image_to_string(bin_normal_inv, config=config_fast)
        ocr_logs.append({
            "roi": roi_name,
            "method": "normal_invert",
            "text": text
        })

        st.session_state["tesseract_debug"]["images"].append(
            (f"OCR前処理（通常）{roi_name}", bin_normal)
        )
        st.session_state["tesseract_debug"]["images"].append(
            (f"OCR前処理（通常・反転）{roi_name}", bin_normal_inv)
        )

        # --- スプレー缶用は赤帯ありの時だけ ---
        if red_ratio > 0.01:
            bin_spray = preprocess_spray(roi)
            text = pytesseract.image_to_string(
                bin_spray, config=config_fast
            )
            ocr_logs.append({
                "roi": roi_name,
                "method": "spray",
                "text": text
            })

            st.session_state["tesseract_debug"]["images"].append(
                (f"OCR前処理（スプレー）{roi_name}", bin_spray)
            )

            # --- 反転 ---
            bin_inv = cv2.bitwise_not(bin_spray)
            text = pytesseract.image_to_string(
                bin_inv, config=config_fast
            )
            ocr_logs.append({
                "roi": roi_name,
                "method": "invert",
                "text": text
            })

            st.session_state["tesseract_debug"]["images"].append(
                (f"OCR前処理（反転）{roi_name}", bin_inv)
            )

    # -------------------------------------------------
    # OCR結果統合
    # -------------------------------------------------
    RULES = {
        "スプレー缶": ["火気", "高温","注意", "可燃", "ガス", "噴射", "PRESSURIZED"],
        "ライター": ["ライター", "着火", "点火", "可燃性ガス"],
        "乾電池": ["乾電池", "アルカリ", "マンガン", "電池"]
    }

    scores = {k: 0 for k in RULES}

    for log in ocr_logs:
        clean = log["text"].replace(" ", "").replace("\n", "")

        for label, keywords in RULES.items():
            for kw in keywords:
                if kw in clean:
                    scores[label] += 1
                    st.session_state["tesseract_debug"]["keyword_hits"].append({
                        "label": label,
                        "keyword": kw,
                        "roi": log["roi"],
                        "method": log["method"]
                    })

    st.session_state["tesseract_debug"]["scores"] = scores
    st.session_state["tesseract_debug"]["ocr_logs"] = ocr_logs

    # -------------------------------------------------
    # 判定
    # -------------------------------------------------
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score >= 1:
        reason = "Tesseract OCR により警告キーワードを検出"
    else:
        reason = "Tesseract OCR を実行したが有効な警告キーワードは検出されなかった"

    return {
        "label": best_label,
        "reason": reason,
    }
