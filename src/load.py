from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(data_dir="../data/"):
    mndata = MNIST(data_dir)
    X_train, y_train = map(np.array, mndata.load_training())
    X_test, y_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return (X_train, y_train), (X_test, y_test)


def plot_digit(image, label):
    """Plot a single MNIST image."""
    image = image.reshape(28, 28)
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")
    plt.title(label)


def plot_random_digits(images, labels, n=10, x=5, y=2, figsize=4):
    """Plot a random MNIST digit."""
    index = random.sample(range(len(images)), n)
    fig, axs = plt.subplots(y, x, figsize=(figsize * x, figsize * y))
    for i, ax in enumerate(axs.flat):
        ax.imshow(
            images[index[i]].reshape(28, 28), cmap="Greys", interpolation="nearest"
        )
        ax.axis("off")
        ax.set_title(labels[index[i]])
    plt.show()
