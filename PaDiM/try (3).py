import torch
from torchvision import transforms as T
from PIL import Image
import os
import csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# resize = (256, 256)
# cropsize = (150, 300)
#
# transform_x = T.Compose([T.CenterCrop(cropsize),
#                         T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
#                         T.ToTensor(),
#                         T.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])])
#
# x_path = "black.png"
# x_image = Image.open(x_path).convert('RGB')  # RGB形式の画像を読み込む
# x_tensor = transform_x(x_image)
# print(x_tensor)

x = []
extension = ".jpg"
# img_dir = os.path.join(self.dataset_path, self.class_name, phase)
img_dir = r"D:\python\DeepLearning\PaDiM-Anomaly-Detection-Localization-master\dataset\mvtec_anomaly_detection\carpet_s\test"
img_dir = os.path.dirname(os.path.abspath(__file__)) + '/dataset/mvtec_anomaly_detection' + "/carpet_s/test"

# img_dir = r"D:\python\DeepLearning\PaDiM-Anomaly-Detection-Localization-master\dataset\mvtec_anomaly_detection\carpet_s\train"
# 外側のサブフォルダをソート
outer_subfolders = sorted([f.path for f in os.scandir(img_dir) if f.is_dir()])

# 外側のサブフォルダをループ
for outer_subfolder in outer_subfolders:
    print(outer_subfolder)
    # 内側のサブフォルダをソート
    inner_subfolders = sorted([f.path for f in os.scandir(outer_subfolder) if f.is_dir()])

    # 内側のサブフォルダをループ
    for inner_subfolder in inner_subfolders:
        # print(inner_subfolder)
        # JPGファイルをソートしてリストに追加
        jpg_files = sorted([os.path.join(inner_subfolder, f) for f in os.listdir(inner_subfolder) if f.lower().endswith(extension)])
        x.extend(jpg_files)

print(x)

################
# extensions = (".jpg", ".jpeg")
#
# # フォルダをソート
# subfolders = sorted([f.path for f in os.scandir(img_dir) if f.is_dir()])
#
# file_paths = []
#
# # サブフォルダ内のJPGファイルをリストに追加
# for subfolder in subfolders:
#     jpg_files = sorted([os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.lower().endswith(extensions)])
#     file_paths.extend(jpg_files)
#
# print(file_paths)






#
# file_paths = []
# img_types = sorted(os.listdir(img_dir))
# for img_type in img_types:
#     img_type_dir = os.path.join(img_dir, img_type)
#
#     img_fpath_list = sorted([f.path for f in os.scandir(img_type_dir) if f.is_dir()])
#
#     # 内側のサブフォルダをループ
#     for inner_subfolder in img_fpath_list:
#         # JPGファイルをソートしてリストに追加
#         img_fpath_list = sorted([os.path.join(inner_subfolder, f) for f in os.listdir(inner_subfolder) if f.lower().endswith(extension)])
#         x.extend(img_fpath_list)
#
# print(x)



# img_types = sorted(os.listdir(img_dir))
# # print(img_types)
# for img_type in img_types:
#
#     # load images
#     img_type_dir = os.path.join(img_dir, img_type)
#     print(img_type_dir)
#     # print(img_type_dir)
#     # img_date = sorted(os.listdir(img_type_dir))
#     # print(img_date)
#     if not os.path.isdir(img_type_dir):
#         continue
#     # img_fpath_list = sorted([os.path.join(img_date, f)
#     #                          for f in os.listdir(img_type_dir)
#     #                          if f.endswith(extension)])
#     img_fpath_list = sorted([os.path.join(img_type_dir, f)
#                              for f in os.listdir(img_type_dir)
#                              if f.endswith(extension)])
#     print(img_fpath_list)
#     # x.extend(img_fpath_list)
