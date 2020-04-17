import numpy as np
import os

def load_mnist(path, kind='train'):
    import struct
    images_path = os.path.join(path,"{}-images.idx3-ubyte").format(kind)
    labels_path = os.path.join(path,"{}-labels.idx1-ubyte").format(kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    # from IPython import embed; embed()
    # images = norm_mean_std(images)
    return images, labels

def norm_mean_std(img,mean=.5,std=.5):
    img = img/255
    img = (img-mean)/std
    return img


if __name__ == '__main__':
    train_images, train_labels = load_mnist('./data/mnist')
    test_images, test_labels = load_mnist('./data/mnist', 't10k')
    print(train_labels.shape)
    print(train_images[0,:784])