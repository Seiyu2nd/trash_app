# import streamlit as st
# from PIL import Image 

#st.title("札幌市東区ごみ収集カレンダー")
# st.caption("")

# image = Image.open(r"C:\Garbage_sorting\trash_app\pages\sapporo_gomi.png") # 相対パスを指定

# st.image(r"C:\Garbage_sorting\trash_app\pages\sapporo_gomi.png", use_container_width=True) # ページ幅に合わせて画像の大きさ表示

import os
import streamlit as st



# IMAGE_PATH = "pages/sapporo_gomi.png"
# st.image(IMAGE_PATH, width='stretch')
""""
st.title('札幌市家庭ごみ収集カレンダー')
 
# タブを作成する
tab_titles = ['データの前処理', 'モデルのトレーニング', 'モデルの評価', '結果の可視化']
tabs = st.tabs(tab_titles)

# 各タブにコンテンツを追加する
with tabs[0]:
    st.header('データの前処理')
    st.write('ここでデータの前処理を行います...')
 
with tabs[1]:
    st.header('モデルのトレーニング')
    st.write('ここでモデルのトレーニングを行います...')
 
with tabs[2]:
    st.header('モデルの評価')
    st.write('ここでモデルの評価を行います...')
 
with tabs[3]:
    st.header('結果の可視化')
    st.write('ここで結果の可視化を行います...')
"""

import os
import json
import streamlit as st

# =========================================================
# ユーザー設定の読み書き（超簡易版：端末ごとに保存）
# =========================================================
SAVE_FILE = "user_pref.json"

def load_pref():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"district": None, "area": None}

def save_pref(district, area):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"district": district, "area": area}, f, ensure_ascii=False)


# =========================================================
# データ定義（あなたの画像に合わせて編集してください）
# =========================================================
district_list = [
    "中央区", "北区", "東区", "白石区",
    "厚別区", "豊平区", "清田区", "南区",
    "西区", "手稲区"
]

# 区ごとの対応エリア（例）
area_list = {
    "中央区": ["南1条西1丁目", "南1条西2丁目"],
    "北区": ["北10条西5丁目", "北10条西6丁目"],
    "東区": ["北7条東5丁目"],
    # ... 必要に応じて追加 ...
}


# =========================================================
# Streamlit UI
# =========================================================
st.title("札幌市 家庭ごみ収集カレンダー")

# --- 保存された設定を読み込み ---
pref = load_pref()

# --- 区選択 ---
district = st.selectbox(
    "区を選択してください",
    district_list,
    index=district_list.index(pref["district"]) if pref["district"] in district_list else 0
)

# --- 区に対応するエリアの取得 ---
areas = area_list.get(district, [])
area = st.selectbox(
    "地域（条・丁目）を選択してください",
    areas,
    index=areas.index(pref["area"]) if pref["area"] in areas else 0
)

# --- 保存ボタン ---
if st.button("この地域をデフォルトとして保存"):
    save_pref(district, area)
    st.success("保存しました（次回起動時のデフォルトになります）")

# --- カレンダー画像のパス ---
img_path = f"calendars/{district}/{area}.png"

# --- 画像が存在する場合のみ表示 ---
if os.path.exists(img_path):
    st.image(img_path, caption=f"{district} {area} の収集カレンダー", use_container_width=True)
else:
    st.warning("カレンダー画像が見つかりませんでした。")
