import os
import numpy as np
import sys
import glob
from PIL import Image
import cv2

# searchdir = os.path.dirname(os.path.abspath(__file__))
searchdir = os.getcwd()
os.chdir(searchdir)

dirlist = glob.glob(searchdir+ "/*/")
foldername = [os.path.basename(d.rstrip('\\')) for d in dirlist]
print(foldername)

parent = os.path.basename(searchdir)


def dirname(fullpath, n):
    path = fullpath
    for _ in range(n):
        path = os.path.split(path)[0]        
    return os.path.basename(path)

child = dirname(searchdir, 0)
print(child)

hoge = "diff_{}".format(child)
try:
    foldername.remove(hoge)
except:
    pass

os.makedirs(hoge, exist_ok=True)
os.chdir(hoge)

def crop_area(img):
    image = cv2.imread(img)
    # image=cv2.resize(image,(510,340))ss
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    half_w = int(3852/2)
    half_h = int(2408/2)
    x,y,w,h = cv2.boundingRect(max_contour)
    # cropped_rectangle = image[y:y+h, x:x+w]
    top = y + int(h/2 ) - half_h
    bottom = y + int(h/2) +  half_h
    left = x + int(w/2) - half_w
    right = x + int(w/2)+ half_w
    # cropped_rectangle = image[top:bottom, left:right]
    # print(bottom - top, right - left)
    area = (left, top, right, bottom)

    # cv2.imshow('binary', binary)
    # cv2.imshow('cropped_rectangle', cropped_rectangle)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    return area #(left, top, right, bottom)
    
for j in range(len(foldername)):
# for j in range(1):
    
    # os.chdir(foldername[j])
    
    files = sorted([os.path.join(searchdir, foldername[j], f)
                    for f in os.listdir(os.path.join(searchdir, foldername[j]))
                    if f.endswith(".jpg")])
    
    print(foldername[j])
    
    # img = Image.open(files[1])
    # img_resize = img.resize((510, 340))
    # imgname = os.path.basename(files[1])
    # img_resize.save(imgname)
    # print(imgname)
    
    # os.chdir('../')
    # image1 = crop(files[0])
    # image2 = crop(files[1])

    
    
    image1 = Image.open(files[0])
    image2 = Image.open(files[1])
    
    # area = (510, 420, 4872 - 510, 3248 - 420) #(left, top, right, bottom)
    area = crop_area(files[0])
    
    image1 = image1.crop(area)
    image2 = image2.crop(area)
    
    
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    im1_u8 = np.array(image1)
    im2_u8 = np.array(image2)

    if im1_u8.shape != im2_u8.shape:
        print("サイズが違います")
        sys.exit()

    im1_i16 = im1_u8.astype(np.int16)
    im2_i16 = im2_u8.astype(np.int16)

    diff_i16 = im1_i16 - im2_i16


    emphasis = 5
    # emphasis = 15
    # if ipt == 1:
    #     calc = (diff_i16*emphasis + 128) 
    # elif ipt == 2:
    #     calc = (diff_i16 + 10) * emphasis

    # calc = (diff_i16*emphasis + 128) #ダメっぽい
    calc = (diff_i16 + 35) * emphasis
    diff_n_i16 = calc


    diff_u8 = diff_n_i16.astype(np.uint8)

    n = 7
    src_median = cv2.GaussianBlur(diff_u8, (n,n), 2)
    
    diff_u8 = src_median

    diff_img = Image.fromarray(diff_u8)
    
    imgname = "{:0>3}.jpg".format(foldername[j])
    print(imgname)
    print("diff_i16.max",diff_i16.max())
    print("diff_i16.min",diff_i16.min())    
    print(diff_n_i16.max())
    print(diff_n_i16.min())   
    
    print("\n")
    diff_img.save(imgname)