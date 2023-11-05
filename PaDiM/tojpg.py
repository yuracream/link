from PIL import Image
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# カレントディレクトリ内のすべてのファイルを取得
files = os.listdir()

# PNGファイルをJPGに変換
for file in files:
    if file.endswith('.png'):
        # 画像を開く
        img = Image.open(file)
        # ファイル名を変更して保存（拡張子をpngからjpgに変更）
        new_file = os.path.splitext(file)[0] + '.jpg'
        print(new_file)
        img.save(new_file)
        # 元のPNGファイルを削除する場合は次の行をアンコメントしてください
        os.remove(file)
