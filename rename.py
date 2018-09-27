import os
import shutil

path1 = '/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/src'
path2 = '/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/label'
path3 = '/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/label1'
path4 = '/home/zq/dataset/Massachusetts_Roads_Dataset/train_data/src1'

pics = os.listdir(path1)
labels = os.listdir(path2)
print(type(labels))
result = []
for i in range(len(pics)):
    name1 = pics[i][:-1]
    num = labels.index(name1)
    result.append(num)
    os.chdir(path1)
    os.rename(pics[i],str(i))
os.chdir(path2)
for r in range(len(result)):
    n = result[r]
    name = labels[n]
    os.rename(name,str(r))