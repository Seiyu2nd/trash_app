import streamlit as st
from PIL import Image
import numpy as np
import os
import pandas as pd
import cv2

# ==== OCR エンジン ====
from ocr_engines.tesseract_engine import run_tesseract
from ocr_engines.easyocr_engine import run_easyocr
from ocr_engines.chatgpt_engine import run_chatgpt_ocr
from ocr_engines.cloudvision_engine import run_cloudvision_multimodal
# ==== MobileNetV2 ====
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory

# ==== 翻訳辞書 ====
from my_translation import translations
from disposal.label_translations import LABEL_TRANSLATIONS
from area_map import AREA_MAP


# ==== 分別ルール ====
from disposal.how_rules import HOW_RULES, DEFAULT_HOW
from disposal.schedule_rules import SCHEDULE_RULES, DEFAULT_SCHEDULE
from disposal.label_normalization import normalize_label 

from disposal.calendar_images import get_calendar_images

# ---------------------------------------------------------
# Google Cloud Vision
# ---------------------------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"C:\Garbage_sorting\trash_app\ocr_engines\formidable-rune-481315-i1-db132fbd239c.json"
)

# ---------------------------------------------------------
# 定数
# ---------------------------------------------------------
MODEL_PATH = "final_trash_model.h5"
TRAIN_DIR = "train"
OCR_TARGET_CLASSES = {"battery", "lighter", "spray","light","box_cutter","scissors"}

JP_LABELS = {
    "battery": "乾電池",
    "lighter": "ライター",
    "spray": "スプレー缶",
    "scissors": "ハサミ",
    "box_cutter": "カッターナイフ",
    "light": "電球"
}

# ---------------------------------------------------------
# Session State 初期化（FSM）
# ---------------------------------------------------------
for key, default in {
    "step": "idle",
    "uploaded_img": None,
    "result": None,
    "is_ambiguous": False,
    "top_label": None,
    "run_sub_ocr": False,
    "ward": None,
    "area": None,
    "second_label": None,

}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------
# 翻訳設定
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("🌏 言語 / Language")

    available_languages = {"日本語": "ja", "English": "en", "中文簡体": "cn", "한국어": "kr"}
    lang_display = st.selectbox("", available_languages.keys())

lang_code = available_languages[lang_display]
t = translations[lang_code]

# ---------------------------------------------------------
# 地域選択
# ---------------------------------------------------------
with st.sidebar:
    st.subheader(t["region_title"])

    wards = list(AREA_MAP[lang_code].keys())

    ward = st.selectbox(
        t["select_ward"],
        wards
    )

    area = st.selectbox(
        t["select_area"],
        AREA_MAP[lang_code][ward]
    )

    st.session_state.ward = ward
    st.session_state.area = area


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title(t["title"])

engine = st.selectbox(
    t["select_ocr"],
    ["MobileNetV2", "ChatGPT-5.1", "CloudVision"]
)

input_method = st.radio(
    t["image_input_method"],
    [t["from_file"], t["from_camera"]]
)

# ---------------------------------------------------------
# 画像入力
# ---------------------------------------------------------
uploaded_img = None

if input_method == t["from_file"]:
    file = st.file_uploader(t["upload_prompt"], ["png", "jpg", "jpeg"])
    if file:
        uploaded_img = Image.open(file)

elif input_method == t["from_camera"]:
    cam = st.camera_input(t["camera_prompt"])
    if cam:
        uploaded_img = Image.open(cam)

if uploaded_img:
    st.session_state.uploaded_img = uploaded_img
    st.image(uploaded_img, use_container_width=True)

# ---------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------
@st.cache_resource
def load_mobilenet():
    model = load_model(MODEL_PATH)
    dataset = image_dataset_from_directory(
        TRAIN_DIR, image_size=(224, 224), shuffle=False
    )
    return model, dataset.class_names

# ---------------------------------------------------------
# 実行
# ---------------------------------------------------------
if uploaded_img and st.button(t["run_ocr"]):
    st.session_state.step = "running"

# ---------------------------------------------------------
# 推論フェーズ
# ---------------------------------------------------------
if st.session_state.step == "running":
    st.session_state.step = "processing"

    if engine == "ChatGPT-5.1":
        with st.spinner(t["ChatGPT_Running"]):
            st.session_state.result = run_chatgpt_ocr(st.session_state.uploaded_img)
        st.session_state.step = "ocr_done"

    elif engine == "CloudVision":
        with st.spinner(t["CloudVision_Running"]):
            st.session_state.result = run_cloudvision_multimodal(st.session_state.uploaded_img)
        st.session_state.step = "ocr_done"

    elif engine == "MobileNetV2":
        with st.spinner(t["MobileNetV2_Running"]):
            model, class_names = load_mobilenet()

            img = st.session_state.uploaded_img.convert("RGB").resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            pred = model.predict(x)[0]

        sorted_probs = sorted(
            zip(class_names, pred),
            key=lambda x: x[1],
            reverse=True
        )

        top_class, top_prob = sorted_probs[0]
        second_class, second_prob = sorted_probs[1]

        jp_label = JP_LABELS.get(top_class, top_class)
        display_label = (
            LABEL_TRANSLATIONS
            .get(lang_code, {})
            .get(jp_label, jp_label)
        )

        st.success(
            f"{t['predicted_label']}：{display_label}（{top_prob*100:.1f}%）"
        )
        # ===== 認識率グラフ（上位5クラス）=====
        top_k = 6
        top_items = sorted_probs[:top_k]

        labels = []
        scores = []

        for cls, prob in top_items:
            jp = JP_LABELS.get(cls, cls)
            display = (
                LABEL_TRANSLATIONS
                .get(lang_code, {})
                .get(jp, jp)
            )
            labels.append(display)
            scores.append(prob * 100)

        chart_df = pd.DataFrame(
            {"認識率(%)": scores},
            index=labels
        )

        st.subheader(t["confidence_chart"])
        st.bar_chart(chart_df)

        # ===== あいまい度の数値化 =====
        ambiguity_score = 1.0 - (top_prob - second_prob)
        ambiguity_score = max(0.0, min(1.0, ambiguity_score))
        ambiguity_score = float(ambiguity_score)  

        # 保存（後段でも使える）
        st.session_state.ambiguity_score = ambiguity_score

        # ===== あいまい度の表示 =====
        st.subheader(t["ambiguity_score"])
        st.write(f"{ambiguity_score * 100:.1f}%")
        st.progress(ambiguity_score)

        # メッセージ表示
        if ambiguity_score < 0.3:
            st.success(t["confidence_high"])
        elif ambiguity_score < 0.6:
            st.warning(t["confidence_medium"])
        else:
            st.error(t["confidence_low"])


        # ===== あいまい判定（既存ロジックと統合） =====
        is_ambiguous = ambiguity_score >= 0.3

        st.session_state.top_label = jp_label
        st.session_state.second_label = JP_LABELS.get(second_class, second_class)

        st.session_state.is_ambiguous = (
            is_ambiguous and
            top_class in OCR_TARGET_CLASSES and
            second_class in OCR_TARGET_CLASSES
        )


        # ===== 次のステップ =====
        if not st.session_state.is_ambiguous:
            # 明確 → 即確定
            st.session_state.result = {
                "label": jp_label,
                "reason": t["auxiliary_OCR_reason"]
            }
            st.session_state.step = "ocr_done"
        else:
            # 曖昧 → 第二候補確認へ
            st.session_state.step = "confirm_second"


# ---------------------------------------------------------
# 第二候補 確認フェーズ
# ---------------------------------------------------------
if st.session_state.step == "confirm_second":

    second_display = (
        LABEL_TRANSLATIONS
        .get(lang_code, {})
        .get(st.session_state.second_label, st.session_state.second_label)
    )

    st.warning(
        t["second_candidate_question"].format(
            label=second_display
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(t["yes"]):
            st.session_state.result = {
                "label": st.session_state.second_label,
                "reason": t["second_candidate_reason"]
            }
            st.session_state.step = "ocr_done"
            st.rerun()

    with col2:
        if st.button(t["no"]):
            st.session_state.step = "classified"
            st.rerun()

    with col3:
        if st.button(t["unknown"]):
            st.session_state.step = "classified"
            st.rerun()


# ---------------------------------------------------------
# 補助OCR選択
# ---------------------------------------------------------
if st.session_state.step == "classified":
    st.warning(t["aimai_hantei"])

    st.selectbox(
        t["auxiliary_OCR"],
        [t["not_use"], "Tesseract", "EasyOCR"],
        key="sub_ocr_select"
    )

    if st.button(t["auxiliary_OCR_run"]):
        st.session_state.run_sub_ocr = True

# ---------------------------------------------------------
# 補助OCR 実行
# ---------------------------------------------------------
if st.session_state.run_sub_ocr:
    st.session_state.run_sub_ocr = False

    img = st.session_state.uploaded_img
    engine = st.session_state.sub_ocr_select

    with st.spinner(t["auxiliary_OCR_Running"]):
        if engine == "Tesseract":
            result = run_tesseract(img)
        elif engine == "EasyOCR":
            result = run_easyocr(img)
        else:
            result = {
                "label": st.session_state.top_label,
                "reason": t["auxiliary_OCR_reason"]
            }

    st.session_state.result = result
    st.session_state.step = "ocr_done"
    st.rerun()

# ---------------------------------------------------------
# 結果表示 + 分別
# ---------------------------------------------------------
if st.session_state.step == "ocr_done":
    result = st.session_state.result

    raw_label = result.get("label")
    label = normalize_label(raw_label)

    display_label = (
        LABEL_TRANSLATIONS
        .get(lang_code, {})
        .get(label, label)
    )

    st.subheader(t["predicted_label"])
    st.success(display_label)

    if "reason" in result:
        st.subheader(t["reasoning"])
        st.write(result["reason"])

    # 分別（共通）
    how_info = (
        HOW_RULES
        .get(lang_code, {})
        .get(label, DEFAULT_HOW)
    )

    st.subheader(t["sorting_method"])
    st.write(f"{t['sorting_classification']}{how_info['category']}")
    st.write(f"{t['How_to_Dispose']}{how_info['how']}")
    st.write(f"{t['Caution']}{how_info['notice']}")
    st.write(f"{t['Cost']}{how_info['commission']}")

    # 回収日（地域依存）
    schedule_map = SCHEDULE_RULES.get(
        (st.session_state.ward, st.session_state.area), {}
    )
    schedule = schedule_map.get(
        how_info["category"],
        DEFAULT_SCHEDULE
    )

    st.subheader(t["Collection_date"])
    st.write(schedule)

    # -------------------------------
    # カレンダー画像表示
    # -------------------------------
    st.subheader(t["Calendar"])

    calendar_images = get_calendar_images(
        st.session_state.ward,
        st.session_state.area
    )

    if calendar_images:
        for img_path in calendar_images:
            st.image(img_path, use_container_width=True)
    else:
        st.info(t["error_Calendar"])

    # =====================================================
    #  OCR デバッグ情報（Tesseract 使用時のみ）
    # =====================================================
    if "tesseract_debug" in st.session_state:

        dbg = st.session_state["tesseract_debug"]

        with st.expander("🔧 OCRデバッグ情報（開発者向け）", expanded=False):

            st.subheader("OCR 前処理・ROI 画像")

            for title, img in dbg["images"]:
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img.ndim == 3 else img,
                    caption=title,
                    use_container_width=True
                )

            st.subheader("キーワード検出内訳")
            if dbg["keyword_hits"]:
                st.table(dbg["keyword_hits"])
            else:
                st.info("キーワードは検出されませんでした")

            st.subheader("スコア集計")
            st.write(dbg["scores"])
