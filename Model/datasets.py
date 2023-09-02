
import torch
import numpy as np
import glob as glob
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def create_npdataset(path_patches, path_plate):
    img_label_0 = glob.glob(path_patches + r"/*")
    img_label_1 = glob.glob(path_plate + r"/*")
    img_path = img_label_0 + img_label_1

    label = [0] * len(img_label_0) + [1] * len(img_label_1) # create label: 0 for non-plate, 1 for plate

    imgList = []
    for i in tqdm(range(len(img_path))):
        img = cv2.imread(img_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(-1).T
        imgList.append(img)

    X = np.array(imgList)
    y = np.array(label)

    return (X, y)

def create_loader(X, y, test_size, batch_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    X_trainL = []
    for i in range(X_train.shape[0]):
        X_trainL.append(X_train[i].reshape(50, 200, order = "C"))
    X_trainL = np.array(X_trainL).reshape(-1, 1, 50, 200)

    X_testL = []
    for i in range(X_test.shape[0]):
        X_testL.append(X_test[i].reshape(50, 200, order = "C"))
    X_testL = np.array(X_testL).reshape(-1, 1, 50, 200)

    print(X_trainL.shape, X_testL.shape, y_train.shape, y_test.shape)
    X_train = torch.from_numpy(X_trainL).float()
    y_train = torch.from_numpy(y_train).float()
    train_Dataset = TensorDataset(X_train, y_train)

    X_test = torch.from_numpy(X_testL).float()
    y_test = torch.from_numpy(y_test).float()
    test_Dataset = TensorDataset(X_test, y_test)

    train_Loader = DataLoader(train_Dataset, batch_size = batch_size, shuffle = True)
    test_Loader = DataLoader(test_Dataset, batch_size = len(y_test), shuffle = False)
    

    return (train_Loader, test_Loader)


X, y = create_npdataset("../training image/patches", "../training image/plate_for_train")


train_Loader, test_Loader = create_loader(X, y, 0.2, 64)
torch.save(train_Loader, "../DataLoader/train_Loader.pt")
torch.save(test_Loader, "../DataLoader/test_Loader.pt")

