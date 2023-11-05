import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

resize = (256,256)
# resize = 224
# resize = 112
# cropsize = 112
cropsize = 224
cropsize = (150,300)

# transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
# transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
transform_x = T.Compose([T.CenterCrop(cropsize),
                        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
# transform_x = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

# x_path = "000.png"
x_path = "black.png"
# x_path = "000.jpg"
x_image = Image.open(x_path).convert('RGB')
x_tensor = transform_x(x_image)
print(x_tensor.size())

# 画像の表示
plt.imshow(x_tensor.permute(1, 2, 0).numpy())  # テンソルをNumPyに変換して表示
print(x_tensor.size())
# plt.title("ANTIALIAS")
# plt.title("LANCZOS")
# plt.title("BILINEAR")
plt.title("BICUBIC")
# plt.show()
