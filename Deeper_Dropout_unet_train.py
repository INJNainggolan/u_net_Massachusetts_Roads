# coding=utf-8
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.models import *
from keras.optimizers import *
from keras.layers.merge import concatenate
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
np.random.seed(seed)

# data_shape = 360*480
img_w = 128
img_h = 128
# 有一个为背景
# n_label = 4+1
n_label = 1

classes = [0., 1.]

adam_I = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-08)

labelencoder = LabelEncoder()
labelencoder.fit(classes)

#image_sets = ['1.png', '2.png', '3.png']


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img


filepath = '/home/zq/dataset/Massachusetts_Roads_Dataset/train/'


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

            # data for validation


def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def unet():
    inputs = Input((img_w, img_h, 3))  # maxl

    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), drop4], axis=3)  # maxl
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)  # maxl
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)  # maxl
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)  # maxl
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=adam_I, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    EPOCHS = 15  # maxl
    BS = 8  # batch_size
#    model = unet()
    model = load_model('/home/zq/3rd_unet_M_epoch_15_bs_8_image_10000_Deeper_Dropout.h5')
    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                            callbacks=callable, max_q_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('/home/zq/output/unet_Massachusetts_Roads/unet_Massachusetts_Roads.png')


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",
                    default=True)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    filepath = '/home/zq/dataset/Massachusetts_Roads_Dataset/train/'
    train(args)
    # predict()