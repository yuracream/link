import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 事前に学習されたResNet18モデルのロード
model = resnet18(pretrained=True)
print(model)
model.eval()
#
# # 画像の前処理とリサイズ後の画像表示
# def preprocess_and_display(image_path):
#     # 画像読み込み
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_image_tensor = transform(image).unsqueeze(0)
#
#     # リサイズ後の画像表示
#     resized_image = input_image_tensor.squeeze(0).permute(1, 2, 0).numpy()
#     resized_image = (resized_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
#     resized_image = resized_image.astype(np.uint8)
#
#     cv2.imshow('Resized Image', resized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return input_image_tensor
#
# # 画像の前処理とリサイズ後の画像表示
# image_path = '004.png'
# input_image_tensor = preprocess_and_display(image_path)
#
# # 画像を分類
# def classify_image(model, image_tensor):
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
#         return probabilities[1].item()  # 1番目のクラスが矩形の存在を表すと仮定
#
# # 画像を分類
# probability = classify_image(model, input_image_tensor)
#
# # 矩形が存在するかの判定
# threshold = 0.5  # 任意の確率の閾値
# is_rectangle_present = probability > threshold
#
# # 矩形が存在する場合、位置をimshow
# if is_rectangle_present:
#     # 仮の処理 - 簡単な方法として、画像全体を表示することにします
#     image = cv2.imread(image_path)
#     cv2.imshow('Detected Rectangle', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print('No rectangle detected.')
