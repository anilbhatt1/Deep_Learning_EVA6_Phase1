import numpy as np

class stats_collector():
    def __init__(self):
        self.losses = []
        self.accuracy = []
        self.img = []
        self.pred = []
        self.label = []

    def append_loss(self, loss):
        self.losses.append(loss)

    def append_acc(self, acc):
        self.accuracy.append(acc)

    def append_img(self, image):
        self.img.append(image)

    def append_pred(self, pred):
        self.pred.append(pred)

    def append_label(self, label):
        self.label.append(label)


class unnorm_img():
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def unnorm_rgb(self, img):
        img = img.numpy().astype(dtype=np.float32)

        for i in range(img.shape[0]):
            img[i] = (img[i] * self.stdev[i]) + self.mean[i]

        return np.transpose(img, (1, 2, 0))

    def unnorm_gray(self, img):
        img = img.numpy().astype(dtype=np.float32)
        img = (img * self.stdev) + self.mean

        return img