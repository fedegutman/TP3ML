import numpy as np

def one_hot_encoding(y:np.ndarray):
    rows = len(y)
    columns = max(y) + 1
    label_matrix = np.zeros((rows, columns), dtype=int)

    for i in range(rows):
        label_matrix[i][y[i]] = 1
    
    return label_matrix

def normalize(array):
    return array/255
