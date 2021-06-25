import numpy as np
import matplotlib.pyplot as plt

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

    def unnorm_albumented(self, img):
        for i in range(img.shape[0]):
            img[i] = (img[i]*channels_stdev[i])+channels_mean[i]
            img = img.permute(1, 2, 0)
        return img

class plots():

    # 5 * 5 images
    def plot_misclassified(self, test_stats, num_images, title, class_names, fig_size, unnorm_mispred):

        figure = plt.figure(figsize=fig_size)
        print(f'** Plotting misclassified images from last epoch for {title} **')
        print('\n')
        class_names_dict = class_names
        if len(test_stats.img) > num_images:
            for i in range(num_images):
                plt.subplot(5, 5, i + 1)
                plt.axis(False)
                unnorm = unnorm_mispred.unnorm_gray(test_stats.img[i].squeeze().cpu())
                plt.imshow(unnorm, cmap='gray_r')
                prediction = class_names_dict.get(test_stats.pred[i])
                actual = class_names_dict.get(test_stats.label[i])
                s = "pred=" + str(prediction) + " act=" + str(actual)
                plt.text(2, -1, s)
        else:
            print(f'Unable to plot - Less than {num_images} images, only have {len(test_stats.img)} images')


counters = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'mis_img':[], 'mis_pred':[], 'mis_lbl':[]}

def ctr():
    def inner(value, type):
        counters[type].append(value)
        return counters
    return inner
