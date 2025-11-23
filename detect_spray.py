# OpenCVで画像処理⇒TesseractでOCR

import cv2
import numpy as np
import pytesseract
from PIL import Image
import streamlit as st
from my_translation import translations


available_languages = {
    '日本語': 'ja',
    'English': 'en'
}

lang_code = available_languages
t = translations[lang_code]

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ==================================
# 白文字抽出：新しい高精度バージョン
# ==================================
def extract_white_text_saturation(roi):
    """赤背景上の白文字を彩度ベースで抽出（最も安定する方法）"""

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # -------------------------------
    # 白文字は「彩度が低い」 → s が小さい
    # ROI 内部の平均彩度値から自動で閾値を作る
    # -------------------------------
    sat_thresh = np.mean(s) * 0.8   # 0.7〜0.9の範囲が最も安定
    _, mask_white = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY_INV)

    # --- ノイズ除去 ---
    mask_white = cv2.medianBlur(mask_white, 3)

    # --- 細文字を強調（OCR精度が劇的に上がる） ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_white = cv2.dilate(mask_white, kernel, iterations=1)

    return mask_white


# ==================================
# メイン関数
# ==================================
def detect_spray_by_text(image_pil):
    """赤帯ROI抽出とOCR精度を最大化した最新安定版"""

    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # --- HSV変換 ---
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # 赤色（2つの領域：0〜10, 160〜179）
    lower_red1 = np.array([0, 60, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 60, 40])
    upper_red2 = np.array([179, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # --- 形態学処理（赤帯領域を安定化） ---
    kernel = np.ones((7, 7), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)

    red_area = cv2.bitwise_and(img_cv, img_cv, mask=mask_red)
    st.image(cv2.cvtColor(red_area, cv2.COLOR_BGR2RGB),
             caption="赤背景抽出", use_container_width=True)

    # --- 輪郭抽出 ---
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / h if h > 0 else 0

        # 条件緩和済み
        if area > 2000 and 2 < aspect < 45:
            rects.append((x, y, w, h))

    st.write(f"赤帯候補として残った矩形数: {len(rects)}")

    # --- 不足時の追加緩和 ---
    if len(rects) == 0:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = w / h if h > 0 else 0
            if area > 1500 and 1.5 < aspect < 60:
                rects.append((x, y, w, h))
        st.write(f"条件緩和後の矩形数: {len(rects)}")

    # --- 縦方向マージ ---
    def merge_vertical(rects, thresh=25):
        if len(rects) <= 1:
            return rects
        rects = sorted(rects, key=lambda r: r[1])
        merged = [rects[0]]

        for r in rects[1:]:
            px, py, pw, ph = merged[-1]
            x, y, w, h = r
            if abs(py - y) < thresh:
                nx = min(px, x)
                nw = max(px + pw, x + w) - nx
                ny = min(py, y)
                nh = max(py + ph, y + h) - ny
                merged[-1] = (nx, ny, nw, nh)
            else:
                merged.append(r)

        return merged

    rects = merge_vertical(rects)
    st.write(f"縦方向マージ後の矩形数: {len(rects)}")

    # --- 最適ROI(赤帯領域)選択 ---
    candidate = None
    max_score = 0
    for x, y, w, h in rects:
        score = w * h * (w / h)
        if score > max_score:
            max_score = score
            candidate = (x, y, w, h)

    if candidate:
        x, y, w, h = candidate
        roi_color = img_cv[y:y+h, x:x+w]
        st.image(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB),
                 caption="選択された赤帯 ROI", use_container_width=True)
    else:
        st.warning("赤帯ROIを特定できませんでした（全体でOCRを実行します）")
        roi_color = img_cv

    # ==================================
    # 🔥 新しい白文字抽出（彩度ベース）
    # ==================================
    bin_img = extract_white_text_saturation(roi_color)

    st.image(bin_img, caption="白文字抽出結果（彩度ベース）", use_container_width=True)

    # --- OCR ---
    config = r'--oem 3 --psm 6 -l jpn'
    text = pytesseract.image_to_string(bin_img, config=config)
    text_clean = text.strip().replace(" ", "")

    st.text_area("OCR認識結果", text_clean if text_clean else "（文字を検出できませんでした）")

    # --- 判定 ---
    keywords = ["火気", "高温", "注意"]
    if any(k in text_clean for k in keywords):
        st.success(t['detect_spray'])
        return True
    else:
        st.info("特定の警告文は検出されませんでした。")
        return False
