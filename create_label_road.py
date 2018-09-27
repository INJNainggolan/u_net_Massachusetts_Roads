import cv2
import numpy as np

if __name__ == '__main__':
    for a in range(1, 76):
        image = cv2.imread('/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/1_75/visualize/{0}.tif'.format(a), cv2.IMREAD_GRAYSCALE)

        print(image)

        print(image.shape)

        #####################################
        # 其他0 黑色 0 0 0 | 0 0 0

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == 255:  # 创建道路数据
                    image[i, j] = 1
                else:
                    image[i, j] = 0

        cv2.imwrite('/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/1_75/label/{0}.png'.format(a), image[:, :])