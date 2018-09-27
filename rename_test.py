import os
path1 = '/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/src'
path2 = '/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/label'
pics = os.listdir(path1)
print(len(pics))
labels = os.listdir(path2)
print(len(labels))