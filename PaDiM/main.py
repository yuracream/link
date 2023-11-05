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
import datasets.mvtec as mvtec

import time
import torchvision.models as models
# import sys
# sys.path.append('D:\python\DeepLearning\PaDiM-Anomaly-Detection-Localization-master/try')
# from mahalanobis_distance import mahalanobis_distance
import csv


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# device setup
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '\dataset\mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '\mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def main():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        # model = wide_resnet50_2(pretrained=True, progress=True)
        # model = wide_resnet50_2(weights='IMAGENET1K_V2', progress=True)
        model = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
        
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    # print("STEP1")
    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        fullpath = test_dataset.x

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            # print("STEP2")
            for (x) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                # print("STEP3")
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            # print("STEP3")
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            # print("STEP4")
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            # print("STEP5")
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        test_imgs = []
        ng_images = {}  # NG画像のファイル名を保存するリスト
        result_dict = {}
        # extract test set features
        # print("STEP6")
        for (x) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            # print("STEP7")
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        # print("STEP8")
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        print('layer1' + ': ', embedding_vectors.size())
        # print("STEP9")
        for layer_name in ['layer2', 'layer3']:
            print(layer_name + ': ', test_outputs[layer_name].size())
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        print("after concat: ", embedding_vectors.size())
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        print("after randomly select: ", embedding_vectors.size())

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        print("BCHW", B,C,H,W)
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        # print("STEP10")
        print("embedding_vectors:",embedding_vectors.shape)#embedding_vectors: (8, 550, 3136)
        print("train_outputs[0]:",train_outputs[0].shape)#train_outputs[0]: (550, 3136)
        print("train_outputs[1]:",train_outputs[1].shape)#train_outputs[1]: (550, 550, 3136)

        # # ブロードキャストを利用してmeanとconv_invを計算
        # mean = train_outputs[0].T  # (3136, 550)
        # conv_inv = np.linalg.inv(train_outputs[1].transpose(2, 0, 1))  # (3136, 550, 550)

        # # embedding_vectorsを変形して距離を計算
        # embedding_vectors_reshaped = embedding_vectors.transpose(2, 1, 0)  # (3136, 550, 8)
        # diff = embedding_vectors_reshaped - mean[:, :, np.newaxis]  # ブロードキャストを使用して差分を計算 (3136, 550, 8)
        # mahalanobis_distances = np.sqrt(np.einsum('ijk,ikj->ij', np.einsum('ijl,jkl->ijk', diff, conv_inv), diff))
        # dist_list.append(mahalanobis_distances)

        for i in range(H * W):
            if i % int(H * W / 10) == 0:
                progress = int((i / int(H * W / 10)) * 10)
                print("{}% complete".format(progress))
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            # dist = [mahalanobis_distance(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        threshold = 0.5

        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, threshold, save_dir, class_name, ng_images, fullpath, result_dict)

    # NG画像のファイル名をCSVファイルに書き込む
    with open(args.save_path + '/' + 'ng_fullpath.csv', 'w', newline='') as csvfile:
        fieldnames = ['NG FULLPATH', 'SCORE']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        for key, value in ng_images.items():
            csvwriter.writerow({'NG FULLPATH': key, 'SCORE': value})

        # csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(['NG FULLPATH'])
        # for item in ng_images:
        #     csvwriter.writerow([item])

        # csvwriter = csv.writer(csvfile)
        # for item in ng_images:
        #     csvwriter.writerow([item])

    # 未ソートの結果をCSVファイルに書き込む
    with open(args.save_path + '/' + 'unsorted_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['ALL FULLPATH', 'SCORE']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        for key, value in result_dict.items():
            csvwriter.writerow({'ALL FULLPATH': key, 'SCORE': value})

        # csvwriter = csv.writer(csvfile)
        # for key, value in result_dict.items():
        #     csvwriter.writerow([key, value])

    # ソートされた結果をCSVファイルに書き込む
    sorted_results = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=False))
    with open(args.save_path + '/' + 'sorted_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['ALL FULLPATH', 'SCORE']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        
        # データを書き込む
        for key, value in sorted_results.items():
            csvwriter.writerow({'ALL FULLPATH': key, 'SCORE': value})

        # csvwriter = csv.writer(csvfile)
        # for key, value in sorted_results.items():
        #     csvwriter.writerow([key, value])

def plot_fig(test_img, scores, threshold, save_dir, class_name, ng_images, fullpath, result_dict):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        mask = scores[i]
        result_dict[fullpath[i]] = mask.max()

        # スコアが閾値を超えたらNG画像のリストに追加
        if mask.max() > 0:
            ng_images[fullpath[i]] = mask.max()
            # ng_images.append(fullpath[i])

        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    start = time.time()
    print("\nSTART\n")

    main()

    end = time.time()
    execution = end - start
    print("\nExecution: {:.2f}sec".format(execution))
    print("\a")
    print("\nEND\n")
    
