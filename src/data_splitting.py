import numpy as np
from src.models import NeuralNetwork
from tqdm import tqdm # type: ignore

def split_dataset(X_dataset:np.ndarray, Y_dataset:np.ndarray, seed:int=42, train_size:float=0.8):
    '''
    '''
    indices = np.arange(len(X_dataset))
    np.random.shuffle(indices)

    split = int(len(indices) * train_size)
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train, X_test = X_dataset[train_indices], X_dataset[test_indices]
    Y_train, Y_test = Y_dataset[train_indices], Y_dataset[test_indices]

    return X_train, X_test, Y_train, Y_test

def cross_validation_grid_search(ModelClass, X, Y, layer_sizes, l2s, optimizers, batch_sizes, patiences, dropouts, k_folds=3):

    best_acc = None
    best_params = None
    best_train_loss = None
    best_val_loss = float('inf')

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    for layer_size in layer_sizes:
        for l2 in l2s:
            for batch in batch_sizes:
                for patience in patiences:
                    for dropout in dropouts:
                        for opt in optimizers:
                            val_scores = []
                            train_losses = []
                            val_losses = []

                            if opt == 'SGD':
                                lr = 0.01
                            else:
                                lr = 0.001

                            # Shuffle once per configuration
                            np.random.shuffle(indices)
                            fold_sizes = np.full(k_folds, n_samples // k_folds)
                            fold_sizes[:n_samples % k_folds] += 1
                            current = 0
                            
                            for fold in range(k_folds):
                                start, stop = current, current + fold_sizes[fold]
                                val_idx = indices[start:stop]
                                train_idx = np.concatenate((indices[:start], indices[stop:]))
                                current = stop

                                X_train, Y_train = X[train_idx], Y[train_idx]
                                X_val, Y_val = X[val_idx], Y[val_idx]

                                model = ModelClass(X_train, Y_train, layer_size)
                                model.train(
                                    X_val, Y_val, epochs=200, lr=lr, batch_size=batch,
                                    optimizer=opt, early_stopping=True, l2_lambda=l2, dropout_rate=dropout, print_progress=False, scheduling_type='Linear'
                                )

                                y_pred = model.predict(X_val)
                                acc = np.mean(y_pred == np.argmax(Y_val, axis=1))
                                val_scores.append(acc)
                                train_losses.append(model.train_losses[-1])
                                val_losses.append(model.val_losses[-1])

                            avg_acc = np.mean(val_scores)
                            avg_train_loss = np.mean(train_losses)
                            avg_val_loss = np.mean(val_losses)
                            print(f'layer_sizes={layer_size}, lr={lr}, l2={l2}, batch={batch}, patience={patience}, dropout={dropout}, opt={opt} -> '
                                f'acc={avg_acc:.4f}, train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')

                            if avg_val_loss < best_val_loss:
                                best_acc = avg_acc
                                best_params = (layer_size, lr, l2, batch, patience, dropout, opt)
                                best_train_loss = avg_train_loss
                                best_val_loss = avg_val_loss

    print('\nBest Params:')
    print(f'layers={best_params[0]}, lr={best_params[1]}, l2={best_params[2]}, batch={best_params[3]}, '
          f'patience={best_params[4]}, dropout={best_params[5]}, opt={best_params[6]}')
    print(f'Validation Accuracy: {best_acc:.4f}, Train Loss: {best_train_loss:.4f}, Validation Loss: {best_val_loss:.4f}')

def architecture_cross_val_grid_search(ModelClass, hidden_layers):
    return