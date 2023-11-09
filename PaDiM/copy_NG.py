#!/usr/bin/env python
import os
import csv
import shutil

# 元のCSVファイルとNGフォルダのパスを指定
csv_file_path = "ng_images.csv"
ng_folder_path = "NG"

# NGフォルダが存在しない場合は作成する
if not os.path.exists(ng_folder_path):
    os.makedirs(ng_folder_path)

# CSVファイルから画像のフルパスを読み取り、NGフォルダにコピーする
with open(csv_file_path, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # ヘッダー行をスキップ
    for row in csvreader:
        image_path = row[1]  # 画像のフルパスはCSVの2列目にあると仮定
        # 画像をNGフォルダにコピー
        shutil.copy(image_path, os.path.join(ng_folder_path, os.path.basename(image_path)))

print("画像のコピーが完了しました。")
