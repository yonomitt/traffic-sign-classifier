
# coding: utf-8

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file =   "../data/traffic-signs-data/train.p"
validation_file = "../data/traffic-signs-data/valid.p"
testing_file =    "../data/traffic-signs-data/test.p"
### Preprocessed file
training_pp_file = "../data/traffic-signs-data/train_pp.p"
### Preprocessed balanced file
training_pp_bal_file = "../data/traffic-signs-data/train_pp_bal.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


import cv2
import numpy as np

from collections import Counter
from sklearn.utils import shuffle
from tqdm import tqdm

def scale(img, x):
    rows, cols, ch = img.shape
    M = np.float32([[x, 0, 0],[0, x, 0]])
    return cv2.warpAffine(img, M, (cols, rows))

def translate(img, x, y):
    rows, cols, ch = img.shape
    M = np.float32([[1, 0, x],[0, 1, y]])
    return cv2.warpAffine(img, M, (cols, rows))

def rotate(img, theta):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    return cv2.warpAffine(img, M, (cols, rows))

### Created larger data set by jittering inputs (scaling, translating, rotating)

X_train_new = []
y_train_new = []

images_pbar = tqdm(range(len(X_train)), desc='X_train', unit='images')
for X, y, i in zip(X_train, y_train, images_pbar):
    X_train_new.append(scale(X, 0.9))
    X_train_new.append(scale(X, 1.1))
    X_train_new.append(translate(X, 2, 2))
    X_train_new.append(translate(X, -2, 2))
    X_train_new.append(translate(X, 2, -2))
    X_train_new.append(translate(X, -2, -2))
    X_train_new.append(rotate(X, 15))
    X_train_new.append(rotate(X, 7.5))
    X_train_new.append(rotate(X, 5.0))
    X_train_new.append(rotate(X, -5.0))
    X_train_new.append(rotate(X, -7.5))
    X_train_new.append(rotate(X, -15))
    
    y_train_new.extend([y] * 12)

out_X = np.concatenate((X_train, X_train_new), axis=0)
out_y = np.concatenate((y_train, y_train_new), axis=0)

with open(training_pp_file, 'wb') as pfile:
    pickle.dump({'features': out_X, 'labels': out_y}, pfile, protocol=pickle.HIGHEST_PROTOCOL)

### Create a larger data set by jittering inputs, but keep the classes balanced

X_train_new = []
y_train_new = []

images_pbar = tqdm(range(len(X_train)), desc='X_train', unit='images')
for X, y, i in zip(X_train, y_train, images_pbar):
    X_train_new.append(scale(X, 0.9))
    X_train_new.append(scale(X, 1.1))
    X_train_new.append(translate(X, 2, 2))
    X_train_new.append(translate(X, -2, 2))
    X_train_new.append(translate(X, 2, -2))
    X_train_new.append(translate(X, -2, -2))
    X_train_new.append(rotate(X, 15))
    X_train_new.append(rotate(X, 7.5))
    X_train_new.append(rotate(X, 5.0))
    X_train_new.append(rotate(X, -5.0))
    X_train_new.append(rotate(X, -7.5))
    X_train_new.append(rotate(X, -15))
    
    y_train_new.extend([y] * 12)

class_dict = Counter(y_train)

min_n = len(y_train) * 13
for c, n in class_dict.items():
    if (n * 13) < min_n:
        min_n = n * 13
        
out_X = np.empty((0,) + X_train[0].shape, dtype=X_train.dtype)
out_y = np.array([], dtype=y_train.dtype)

for c in class_dict.keys():
    only_c = np.array([X for X, y in zip(X_train, y_train) if y == c])
    only_c_new = [X for X, y in zip(X_train_new, y_train_new) if y == c]
    only_c_new = shuffle(only_c_new)
    out_X = np.concatenate((out_X, only_c), axis=0)
    out_X = np.concatenate((out_X, only_c_new[:min_n-len(only_c)]))
    out_y = np.concatenate((out_y, [c] * min_n))

with open(training_pp_bal_file, 'wb') as pfile:
    pickle.dump({'features': out_X, 'labels': out_y}, pfile, protocol=pickle.HIGHEST_PROTOCOL)


