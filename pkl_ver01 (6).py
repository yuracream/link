import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18

import time
import torchvision.models as models
import csv
import sys

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
# from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer


CLASS_NAMES = ['img']
# CLASS_NAMES = ['img_small']
endswith_train = '.jpg'

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
        self.x = self.load_dataset_folder()

        self.transform_x = T.Compose([
                                        T.Resize(resize, interpolation=InterpolationMode.LANCZOS),
                                        T.CenterCrop(cropsize),
                                        # T.Grayscale(num_output_channels=1),
                                        T.ToTensor(),
                                        # T.Normalize(mean=[0.485, 0.456, 0.406],
                                        #           std=[0.229, 0.224, 0.225])
                                        ])
        self.data = self.cvt()
        # self.label = 0
    
    def cvt(self):
        for i in range(len(self.x)):
            x = self.x[i]
            # x = Image.open(x).convert('RGB')
            x = Image.open(x).convert('L')
            x = self.transform_x(x)
            
            C, H, W = x.size()
            x= x.view(1,C,H,W)
            if i == 0:
                self.a = x
            else:
                self.a = torch.cat((self.a,x))
        
        self.a = self.a.to('cpu').detach().numpy().copy()
        # print(self.a.shape)
        return self.a
            

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x = []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        
        i = 0
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith(endswith_train)])
            x.extend(img_fpath_list)
            label = np.full(len(img_fpath_list),i)
            try:
                self.label = np.append(self.label, label)
                # print(self.label)
                
                # self.label = np.insert(self.label, label)
            except:
                self.label = label       
            i += 1 
            
        return list(x)

if __name__ == '__main__':
    start = time.time()
    print("\nSTART\n")
    
    
    path = r"C:\Users\????????????????????\ch07"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    
    # x_train = MVTecDataset(path, class_name=CLASS_NAMES[0], is_train=True).data
    # t_train = MVTecDataset(path, class_name=CLASS_NAMES[0], is_train=True).label
    # x_test = MVTecDataset(path, class_name=CLASS_NAMES[0], is_train=False).data
    # t_test = MVTecDataset(path, class_name=CLASS_NAMES[0], is_train=False).label
    # with open("x_train.pkl", "wb") as f:
    #     pickle.dump(x_train, f)
    # with open("t_train.pkl", "wb") as f:
    #     pickle.dump(t_train, f)
    # with open("x_test.pkl", "wb") as f:
    #     pickle.dump(x_test, f)
    # with open("t_test.pkl", "wb") as f:
    #     pickle.dump(t_test, f)
    
    
    with open("x_train.pkl", 'rb') as f:
        x_train = pickle.load(f)
    with open("t_train.pkl", 'rb') as f:
        t_train = pickle.load(f)
    with open("x_test.pkl", 'rb') as f:
        x_test = pickle.load(f)
    with open("t_test.pkl", 'rb') as f:
        t_test = pickle.load(f)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    
    max_epochs = 3

    network = SimpleConvNet(input_dim=(1,224,224), 
                            conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=2, weight_init_std=0.01)
                            
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                    epochs=max_epochs, mini_batch_size=100,
                    optimizer='Adam', optimizer_param={'lr': 0.001},
                    evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # パラメータの保存
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
    

    with open("x_test.pkl", 'rb') as f:
        x_test = pickle.load(f)
    with open("params.pkl", 'rb') as f:
        params = pickle.load(f)
        
    print(x_test.shape)
    # print(params.keys())
    print(params['W1'].shape)
    print(params['b1'].shape)
    print(params['W2'].shape)
    print(params['b2'].shape)
    print(params['W3'].shape)
    print(params['b3'].shape)
    
    
    end = time.time()
    execution = end - start
    print("\nExecution: {:.2f}sec".format(execution))
    print("\a")
    print("\nEND\n")
    