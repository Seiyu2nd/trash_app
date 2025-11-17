from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import base64

# 画像をbase64形式にエンコードする関数
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class BusinessCard(BaseModel):
    person_name_ja: Optional[str]
    person_name_en: Optional[str]
    job_title: Optional[str]
    company: Optional[str]
    email: Optional[str]
    phone_numbers: Optional[str]
    website: Optional[str]
    address: Optional[str]
    
# インプット画像へのパス
image_path = r"C:\Garbage_sorting\trash_app\20250627143418.png"

# APIに入力する画像をbase64形式にエンコード
base64_image = encode_image(image_path)

SYSTEM_PROMPT = """
# Role and Objective
あなたは名刺画像から情報を抽出する、OCRおよび情報抽出の専門エージェントです。  
与えられた名刺画像から必要な情報を構造化データとして正確に抽出することがあなたのミッションです。

# Instructions
- 名刺画像から抽出できる全ての情報を取得してください。
- 指定されたresponse_formatに完全に一致する結果を出力してください。
- **読み取れない、もしくは存在しないフィールドは必ずNoneにしてください**。
- キーの追加・削除・名前変更は禁止です。
- 読み取り項目の出力以外（コメント、説明文、マークダウン、推論過程など）は絶対に含めないでください。

## Sub-categories for more detailed instructions
- 氏名や役職など、明確に判別できるフィールドは可能な限り抽出し、正確に記述してください。
- 連絡先（メールアドレス、電話番号など）は、画像から読み取れる全てを対象にしてください。
- 英語・日本語表記が混在している場合は両方記載してください。
- 番号・URL・住所なども、可読な範囲で正確に記載してください。
- 複数のフィールドが存在する場合は、それぞれのフィールドをカンマで区切って文字列として出力してください。

# Reasoning Steps
1. 画像内の全情報を正確に読み取る。
2. response_formatに沿って各フィールドを正しく抽出する。
3. 抽出できなかった場合はNoneとする。
4. 必要な情報のみをresponse_format通りに出力する。

# Output Format
出力は以下の形式で行ってください。

## Example
{
  "person_name_ja": "山田太郎",
  "person_name_en": "Taro Yamada",
  "job_title": "営業部長",
  "company": "株式会社ABC, ABC Inc.",
  "email": "taro.yamada@example.com",
  "phone_numbers": ["03-1234-5678"],
  "website": "https://www.example.com",
  "address": "東京都千代田区1-2-3"
}

# Context
名刺には日本語・英語で記載された名前や会社名、複数の電話番号や連絡先、住所などが記載されています。
実際の画像内容に忠実に、抜けや漏れがないように情報を抽出してください。

# Final instructions and prompt to think step by step
- 出力前に、あなた自身の中で抽出した情報が要求に完全に合致しているかを必ず確認してください。
- 手順を内部で計画し、思考を整理してから最終出力を行ってください（ただし推論過程は絶対に出力しないこと）。
- 出力はresponse_formatに沿った形式で行ってください。
- 抽出できなかった場合は,Noneとしてください。推測で回答しないこと。
"""

USER_TEXT = "この名刺画像から全ての情報を抽出してください。読み取れない、存在しない項目はNoneにしてください。"


def extract(image_path: str):
    base64_image = encode_image(image_path)

    client = OpenAI()
    messages = [{
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEXT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-2025-04-14",
        messages=messages,
        response_format=BusinessCard,
    )
    output = response.choices[0].message.parsed
    return output

result = extract(image_path)
for key, value in result.model_dump().items():
    print(f"{key}: {value}")