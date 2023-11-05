import os
from PIL import Image
import glob
import shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# 画像のリサイズ後のサイズ
target_size = (4872, 3248)

# 画像が保存されているフォルダのパス
source_folder = "D:\dataset\mvtec_anomaly_detection\grid"
# destination_folder = "D:\dataset\mvtec_anomaly_detection\resize"

# サブフォルダ内のすべてのPNGファイルを取得
png_files = glob.glob(os.path.join(source_folder, '**/*.png'), recursive=True)

# リサイズして保存
for png_file in png_files:
    # try:
    print(png_file)

    # 画像を開く
    img = Image.open(png_file)

    # 画像をリサイズ
    resized_img = img.resize(target_size, Image.ANTIALIAS)

    # # ファイルパスを取得（相対パスから絶対パスに変換）
    # relative_path = os.path.relpath(png_file, source_folder)
    # destination_path = os.path.join(destination_folder, relative_path)
    #
    # # ディレクトリが存在しない場合は作成
    # os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # リサイズした画像を保存
    resized_img.save(png_file)

    # os.remove(png_file)

    # print(f"Resized and saved: {destination_path}")

    # except Exception as e:
    #     print(f"Error processing {png_file}: {str(e)}")
