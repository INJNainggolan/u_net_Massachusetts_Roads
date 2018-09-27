#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 128
img_h = 128

image_sets = ['1.png','2.png','3.png','4.png','5.png',
              '6.png', '7.png', '8.png', '9.png', '10.png',
              '11.png', '12.png', '13.png', '14.png', '15.png',
              '16.png', '17.png', '18.png', '19.png', '20.png',
              '21.png', '22.png', '23.png', '24.png', '25.png',
              '26.png', '27.png', '28.png', '29.png', '30.png',
              '31.png', '32.png', '33.png', '34.png', '35.png',
              '36.png', '37.png', '38.png', '39.png', '40.png',
              '41.png', '42.png', '43.png', '44.png', '45.png',
              '46.png', '47.png', '48.png', '49.png', '50.png',
              '51.png', '52.png', '53.png', '54.png', '55.png',
              '56.png', '57.png', '58.png', '59.png', '60.png',
              '61.png', '62.png', '63.png', '64.png', '65.png',
              '66.png', '67.png', '68.png', '69.png', '70.png',
              '71.png', '72.png', '73.png', '74.png', '75.png']


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 10000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread('/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/1_75/src/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/1_75/label/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            visualize = np.zeros((256,256)).astype(np.uint8)
            visualize = label_roi *50
            
            cv2.imwrite(('/home/zq/dataset/Massachusetts_Roads_Dataset/train/visualize/%d.png' % g_count),visualize)
            cv2.imwrite(('/home/zq/dataset/Massachusetts_Roads_Dataset/train/src/%d.png' % g_count),src_roi)
            cv2.imwrite(('/home/zq/dataset/Massachusetts_Roads_Dataset/train/label/%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1


            
    

if __name__=='__main__':  
    creat_dataset(mode='augment')
