import numpy as np

def dev_test_split(X_dataset:np.ndarray, Y_dataset:np.ndarray, seed:int=42, dev_size:float=0.8):
    '''
    '''
    indices = np.arange(len(X_dataset))
    np.random.shuffle(indices)

    split = int(len(indices) * dev_size)
    dev_indices = indices[:split]
    test_indices = indices[split:]

    X_dev, X_test = X_dataset[dev_indices], X_dataset[test_indices]
    Y_dev, Y_test = Y_dataset[dev_indices], Y_dataset[test_indices]

    return X_dev, X_test, Y_dev, Y_test
    
def train_valid_split(X_dataset:np.ndarray, Y_dataset:np.ndarray, seed:int=42, dev_size:float=0.8):
    '''
    '''
    indices = np.arange(len(X_dataset))
    np.random.shuffle(indices)

    split = int(len(indices) * dev_size)
    train_indices = indices[:split]
    val_indices = indices[split:]

    X_train, X_valid = X_dataset[train_indices], X_dataset[val_indices]
    Y_train, Y_valid = Y_dataset[train_indices], Y_dataset[val_indices]

    return X_train, X_valid, Y_train, Y_valid

def cross_validation():
    return
