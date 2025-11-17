from PIL import Image
import pytesseract

# Tesseractの実行ファイルパス（環境に合わせて変更）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 画像を開く（ここでは引数は "path" だけ！）
img = Image.open(r"C:\Users\you_c\Downloads\sample-ocr.png")

# pytesseractで日本語OCRを実行
text = pytesseract.image_to_string(img, lang='jpn')

print(text)
