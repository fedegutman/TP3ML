import numpy as np
from src.models import NeuralNetwork
from tqdm import tqdm

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


def batch_cross_validation(X_train, y_train, layer_sizes, batch_sizes, k=5, epochs=50, lr=0.01, optimizer='gradient_descent', l2_lambda=0.01):
    results = {}

    for batch_size in tqdm(batch_sizes, leave=False):
        print(f"Performing cross-validation for batch size: {batch_size}")
        fold_size = len(X_train) // k
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        val_losses = []

        for i in range(k):
            # Create validation and training sets for the current fold
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != i])

            X_fold_train, y_fold_train = X_train[train_indices], y_train[train_indices]
            X_fold_val, y_fold_val = X_train[val_indices], y_train[val_indices]

            # Initialize the neural network
            model = NeuralNetwork(X_fold_train, y_fold_train, layer_sizes)

            # Train the model
            model.train(X_fold_val, y_fold_val, epochs=epochs, lr=lr, batch_size=batch_size, optimizer=optimizer, l2_lambda=l2_lambda)

            # Record the validation loss
            val_losses.append(model.val_losses[-1])  # Last validation loss for this fold

        # Store the validation losses for this batch size
        results[batch_size] = val_losses

    return results