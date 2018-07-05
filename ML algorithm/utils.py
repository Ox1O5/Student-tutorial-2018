import os
import struct
import numpy as np

def load_mnist(path, kind='train', flatten = False):
    assert kind == 'train' or kind =='t10k'
    label_file = os.path.join(path, kind + '-labels.idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)
        #new_labels = np.zeros((num, 10)) #(num, 10)
        #new_labels[np.arange(num), labels] = 1  #change label to One-hot
    img_file = os.path.join(path, kind + '-images.idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        imgs = imgs.astype(np.float32)
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, labels

def read_mnist(path, flatten = True):
    train_imgs, train_labels = load_mnist(path, 'train', flatten)
    test_imgs, test_labels = load_mnist(path, 't10k', flatten)

    return train_imgs, train_labels, test_imgs, test_labels