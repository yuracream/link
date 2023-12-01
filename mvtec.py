import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

ireko = False
# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
# CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#                'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# CLASS_NAMES = ['grid']
# CLASS_NAMES = ['grid_1000']
# CLASS_NAMES = ['sub']
CLASS_NAMES = ['FINAL']
# endswith = 'vcom2.jpg'
# endswith_train = '.png'
# endswith_test = 'VCOM_SIMI2.png'
endswith_train = '.jpg'
# endswith_test = 'VCOM_SIMI2.jpg'
endswith_test = '.jpg'

# CLASS_NAMES = ['carpet_s_png']
# endswith = '.png'



# CLASS_NAMES = ['grid_orig']
# resize = (224,224)
# crop = (2500,3900)
resize = (224,224)
crop = (224,224)


class MVTecDataset(Dataset):
    def __init__(self, dataset_path=os.path.dirname(os.path.abspath(__file__)) + '\dataset\mvtec_anomaly_detection', class_name='bottle', is_train=True,
                 resize=resize, cropsize=crop):
                #  resize=256, cropsize=112):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x = self.load_dataset_folder()

        # set transforms
        # self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
        # self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
        self.transform_x = T.Compose([
                                        T.Resize(resize, interpolation=InterpolationMode.LANCZOS),
                                        T.CenterCrop(cropsize),
        # self.transform_x = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        # self.transform_x = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                        ])

    def __getitem__(self, idx):
        x = self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x = []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        if phase == "train" or ireko == False:
            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith(endswith_train)])
                x.extend(img_fpath_list)

        else:
            outer_subfolders = sorted([f.path for f in os.scandir(img_dir) if f.is_dir()])
            for outer_subfolder in outer_subfolders:
                # print(outer_subfolder)
                # 内側のサブフォルダをソート
                inner_subfolders = sorted([f.path for f in os.scandir(outer_subfolder) if f.is_dir()])

                # 内側のサブフォルダをループ
                for inner_subfolder in inner_subfolders:
                    # print(inner_subfolder)
                    # JPGファイルをソートしてリストに追加
                    img_fpath_list = sorted([os.path.join(inner_subfolder, f) for f in os.listdir(inner_subfolder) if f.endswith(endswith_test)])
                    x.extend(img_fpath_list)

            print(x)
            
        return list(x)

#     def download(self):
#         """Download dataset if not exist"""

#         if not os.path.exists(self.mvtec_folder_path):
#             tar_file_path = self.mvtec_folder_path + '.tar.xz'
#             if not os.path.exists(tar_file_path):
#                 download_url(URL, tar_file_path)
#             print('unzip downloaded dataset: %s' % tar_file_path)
#             tar = tarfile.open(tar_file_path, 'r:xz')
#             tar.extractall(self.mvtec_folder_path)
#             tar.close()

#         return


# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
