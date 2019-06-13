import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

class FilterBank:
    def __init__(self):
        self.filter_bank = []
        self.range_dog_filter = range(0, 0)
        self.range_log_filter = range(0, 0)
        self.range_gauss_filter = range(0, 0)
        self.support = 49
        # self.
        self.create_filter()

    def create_filter(self):

        count = 0
        self.filter_bank.extend(self.create_dog_filter())
        length = len(self.filter_bank)
        self.range_dog_filter = range(count, length)
        count = length

        self.filter_bank.extend(self.create_log_filter())
        length = len(self.filter_bank)
        self.range_log_filter = range(count, length)
        count = length

        self.filter_bank.extend(self.create_gaussian_filter())
        length = len(self.filter_bank)
        self.range_gauss_filter = range(count, length)

        self.show_filter_bank()

    def create_gaussian_filter(self):
        filters = []
        sigmas = [1, 2, 4]

        for sigma in sigmas:

            ax = np.arange(-self.support // 2 + 1., self.support // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)

            kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

            kernel = kernel / kernel.sum()

            filters.append(kernel)

        return filters

    def create_log_filter(self):
        filters = []
        sigmas = [1, 2, 4, 8]

        for sigma in sigmas:

            ax = np.arange(-self.support // 2 + 1., self.support // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)

            h = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
            h[h < sys.float_info.epsilon * h.max()] = 0
            h = h/h.sum() if h.sum() != 0 else h
            h1 = h * (np.square(xx) + np.square(yy) - 2 * np.square(sigma)) / (np.power(sigma, 4))

            kernel = h1 - h1.mean()

            filters.append(kernel)

        return filters

    def create_dog_filter(self):
        filters = []
        sigmas = [2, 4]
        scale = 3
        nb_orient = 2

        ax = np.arange(-self.support // 2 + 1., self.support // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax[::-1])

        org_pts = np.vstack((xx.flatten('F'), yy.flatten('F')))

        for sigma in sigmas:
            for i in range(nb_orient):
                angle = np.pi * i / nb_orient
                rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                rot_pts = np.dot(rot_mat, org_pts)

                gx = self.gauss_derivative(0, scale * sigma, 0, rot_pts[0, :])
                gy = self.gauss_derivative(1, sigma, 0, rot_pts[1, :])

                kernel = np.multiply(gx, gy)
                kernel = kernel.reshape((self.support, self.support))
                kernel = kernel - np.mean(kernel[:])
                kernel = kernel/(np.sum(np.abs(kernel)))

                filters.append(kernel)


        return filters


    def gauss_derivative(self, order, sigma, mean, x):
        x = x - mean
        num = np.square(x)
        variance = np.square(sigma)

        g = np.exp(-num/(2*variance))/np.sqrt(np.pi * 2 * variance)
        if order is 1:
            g = np.multiply(-g, x / variance)
        elif order is 2:
            g = np.multiply(g, ((num - variance) / np.square(variance)))

        return g

    def show_filter_bank(self):
        fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(8, 6))
        plt.gray()

        fig.suptitle('Filter bank', fontsize=12)

        # axes[0][0].axis('off')

        for filter, ax in zip(self.filter_bank[:6], axes[0][0:]):
            ax.imshow(filter)
            ax.axis('off')

        axes[0][6].axis('off')

        for filter, ax in zip(self.filter_bank[6:12], axes[1][0:]):
            ax.imshow(filter)
            ax.axis('off')

        axes[1][6].axis('off')

        #for filter, ax in zip(self.filter_bank[12:18], axes[2][0:]):
        #    ax.imshow(filter)
        #    ax.axis('off')

        #axes[2][6].axis('off')

        #for filter, ax in zip(self.filter_bank[18:24], axes[3][0:]):
        #    ax.imshow(filter)
        #    ax.axis('off')

        #axes[3][6].axis('off')

        for filter, ax in zip(self.filter_bank[self.range_log_filter[0]:self.range_gauss_filter[-1]+1], axes[2][0:]):
            ax.imshow(filter)
            ax.axis('off')

        plt.show()

if __name__ == "__main__":
    fb = FilterBank()

