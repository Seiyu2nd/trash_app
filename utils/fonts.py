import os
import urllib.request
import zipfile
import pathlib
import matplotlib.font_manager as fm

def setup_japanese_font():
    font_filename = "ipaexg.ttf"
    if not os.path.isfile(font_filename):
        url = "https://moji.or.jp/wp-content/ipafont/IPAexfont/IPAexfont00401.zip"
        urllib.request.urlretrieve(url, "IPAexfont.zip")
        with zipfile.ZipFile("IPAexfont.zip", "r") as z:
            z.extractall(".")
        os.rename("IPAexfont00401/ipaexg.ttf", font_filename)
    font_path = pathlib.Path(font_filename).resolve()
    jp_prop = fm.FontProperties(fname=str(font_path))
    fm.fontManager.addfont(str(font_path))
    return jp_prop, font_path
