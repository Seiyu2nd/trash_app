import streamlit as st
from PIL import Image 

st.title("札幌市東区ごみ収集カレンダー")
st.caption("")

# image = Image.open(r"C:\Garbage_sorting\trash_app\pages\sapporo_gomi.png") # 相対パスを指定

st.image(r"C:\Garbage_sorting\trash_app\pages\sapporo_gomi.png", use_container_width=True) # ページ幅に合わせて画像の大きさ表示