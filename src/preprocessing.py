import numpy as np
import matplotlib.pyplot as plt

def one_hot_encoding(y:np.ndarray):
    rows = len(y)
    columns = max(y) + 1
    label_matrix = np.zeros((rows, columns), dtype=int)

    for i in range(rows):
        label_matrix[i][y[i]] = 1
    
    return label_matrix

def normalize(array):
    return array/255

def plot_images(images, nrows=2, ncolumns=5):
    fig, ax = plt.subplots(nrows, ncolumns, figsize=(8, 3))
    for i in range(nrows):
        for k in range(ncolumns):
            ax[i, k].imshow(images[np.random.randint(0, len(images))].reshape(28, 28), cmap='grey')
            ax[i, k].axis('off')
    plt.tight_layout()
    plt.show()

def histogram(array, bins):
    plt.hist(array, bins=bins, edgecolor='black', color='blue', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
