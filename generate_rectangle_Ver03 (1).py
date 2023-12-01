from PIL import Image, ImageDraw
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import cv2
import sys
import glob

def pprin(args):
    print("%s: " % args, args)

def draw_rectangle_with_rounded_corners(image_path, output_path):
    # 画像の読み込み
    img = cv2.imread(image_path)
    h, w, _= img.shape
    print("h,w", h, w)
    

    width = np.random.randint(500, int(w/2))
    height = np.random.randint(500, int(h/2))
    # width = 1100
    # height = 750
    x = np.random.randint(-width, w)
    y = np.random.randint(-height, h)
    # x = 1400
    # y = -60
    line_width = np.random.randint(40, 60)
    # line_width = 40
    print("x,y: ",x,y)
    print("width,height: ",width,height)
    print("line_width: ",line_width)
    # 描画オブジェクトの作成
    # draw = ImageDraw.Draw(img)
    split = 4
    min_side = min([width, height])
    r = min_side/split
    
    x2 = max([0,x])
    y2 = max([0,y])
    mean_rect = np.mean(img[y2:y2 + height, x2:x2 + width])
    print("mean_rect: ", mean_rect)
    diff = np.random.randint(2, 15)
    print("diff: ", diff)
    
    for i in range(width):
        for j in range(height):
            if y+j > 0 and x+i > 0:
                # intensity = np.random.randint(165, 185)
                # intensity = np.random.normal(172.67,9.408)   # 平均50、標準偏差10の正規分布
                intensity = np.random.normal(mean_rect - diff, 10)   # 平均50、標準偏差10の正規分布

                if j < line_width or j > height - line_width or i < line_width or i > width - line_width:
                    try:
                        img[y+j, x+i] = [intensity,intensity,intensity]
                    except:
                        pass
                    
                else:
                    # intst = np.random.normal(182.20,7.57)
                    # try:
                    #     img[y+j, x+i] = [intst,intst,intst]
                    # except:
                    #     pass
                        
                    if i < line_width + r and j < line_width + r:
                        xx = i - (line_width + r)
                        yy = j - (line_width + r)
                        if xx**2 + yy**2 > r**2:
                            try:
                                img[y+j, x+i] = [intensity,intensity,intensity]
                            except:
                                pass
                            
                    elif i > width - line_width - r and j < line_width + r:
                        xx = i - (width - line_width - r)
                        yy = j - (line_width + r)
                        if xx**2 + yy**2 > r**2:
                            try:
                                img[y+j, x+i] = [intensity,intensity,intensity]
                            except:
                                pass
                            
                    elif i > width - line_width - r and j > height - line_width - r:
                        xx = i - (width - line_width - r)
                        yy = j - (height - line_width - r)
                        if xx**2 + yy**2 > r**2:
                            try:
                                img[y+j, x+i] = [intensity,intensity,intensity]
                            except:
                                pass
                            
                    elif i < line_width + r and j > height - line_width - r:
                        xx = i - (line_width + r)
                        yy = j - (height - line_width - r)
                        if xx**2 + yy**2 > r**2:
                            try:
                                img[y+j, x+i] = [intensity,intensity,intensity]
                            except:
                                pass
                        
                        
                    pass    
    
    cv2.imwrite(output_path, img)
    

# draw_rectangle_with_rounded_corners("0400.jpg", "output_image3.jpg")

searchdir = os.path.dirname(os.path.abspath(__file__))
# searchdir = os.getcwd()
os.chdir(searchdir)

# dirlist = glob.glob(searchdir+ "/*/")
# foldername = [os.path.basename(d.rstrip('\\')) for d in dirlist]
# print(foldername)

# parent = os.path.basename(searchdir)


def dirname(fullpath, n):
    path = fullpath
    for _ in range(n):
        path = os.path.split(path)[0]        
    return os.path.basename(path)

child = dirname(searchdir, 0)
print(child)

hoge = "aug_{}".format(child)
# try:
#     foldername.remove(hoge)
# except:
#     pass

os.makedirs(hoge, exist_ok=True)
os.chdir(hoge)

files = glob.glob(searchdir+'\*jpg')
i = 0
for file in files:
    print(file)