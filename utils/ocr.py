# OpenAIのGPT-4o-miniで文字を抽出

import base64
from openai import OpenAI
import streamlit as st

@st.cache_data(show_spinner=False)
def extract_text_from_image(uploaded_file):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)

    img_bytes = uploaded_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user",
             "content": [
                 {"type": "text", "text": "この画像に含まれる文字を抽出してください。"},
                 {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
             ]}
        ]
    )
    return response.choices[0].message.content.strip()
