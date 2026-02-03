# ocr_engines/chatgpt_engine.py
import base64
import io
from PIL import Image
import openai
import streamlit as st
import json


def run_chatgpt_ocr(image: Image.Image):
    """
    ChatGPT で OCR + battery/spray/lighter 分類を行い、
    dict 形式で返す。
    """
    try:
        # --- 画像を base64 に変換 ---(Pillowで加工 → Base64 → API)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        prompt = """
                あなたは高度な画像認識＋OCRAIです。
                以下のルールに従い、入力された画像を認識してください。
                - 次の分類のどれかを判定： "battery" , "spray" , "lighter","box_cutter","light","scissors"(label)
                - なぜその分類になったのか理由を説明(reason)
                - ラベルに応じて一般的な日本の自治体の捨て方ガイド（disposal）を生成して返す

                認識した結果、判別に困ったらに迷った場合は、以下のルールに従いOCRを実行してください。
                - 画像内に実在する文字のみ抽出する(text)
                - 画像に写っていない文字は1文字も出力してはいけない
                - 文脈からの推測は禁止
                - 辞書・常識による補完は禁止
                - 出力は「見えた順に近い順序」で行うこと
                - 改行位置も推測してはいけない
                - 認識不能箇所は "§§§" と記載

                出力は JSON **のみ** で返す。

                必ず下記4項目を含み、どれも省略しないこと。空欄にしないこと：
                - text
                - label（"battery" / "spray" / "lighter"/"box_cutter"/"light"/"scissors"/ のいずれか）
                - reason
                - disposal

                返答形式：

                {
                "text": "...",
                "label": "battery|spray|lighter|box_cutter|light|scissors",
                "reason": "...",
                "disposal": "..."
                }

                """


        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "systemから与えられているルールに従い、この画像をOCRして分類してください。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ]
        )

        answer = response.choices[0].message.content

        # ChatGPTが JSON を返すので Python dict に変換
        try:
            result = json.loads(answer)
        except Exception:
            return {
                "text": "",
                "label": None,
                "reason": "[ChatGPT OCR Error] JSON の解析に失敗しました。出力内容: " + str(answer),
                "disposal": ""
            }

        # OCR結果と分類を dict 形式で返す
        return {
            "text": result.get("text", ""),
            "label": result.get("label", None),
            "reason": result.get("reason", ""),
            "disposal": result.get("disposal", "")
        }

    except Exception as e:
        return {
            "text": "",
            "label": None,
            "reason": f"[ChatGPT OCR Error] {e}",
            "disposal": ""
        }
