import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork:
    def __init__(self, X:np.ndarray, y:np.ndarray, layer_sizes:list):
        self.X = X # (samples, features)
        self.Y = y # (samples, classes)
        self.total_samples = y.shape[0]

        self.nlayers = len(layer_sizes) - 1 # sin contar el input
        self.weights = []
        self.biases = []

        self.train_losses = []
        self.val_losses = []

        np.random.seed(42)

        for i in range(self.nlayers):
            std = np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.normal(0, std, size=(layer_sizes[i], layer_sizes[i+1])) # inicializo pesos con glorot
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(W)
            self.biases.append(b)

        # inicializo hiperparametros para adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True)) # keepdims=True para que (n,) -> (n,1)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X, training=True):
        z = [X]
        a = []
        self.dropout_masks = []

        for i in range(self.nlayers - 1): # capas ocultas relu
            W = self.weights[i]
            b = self.biases[i]

            a_l = z[-1] @ W + b
            a.append(a_l)

            z_l = self.relu(a_l)

            # === Dropout ===
            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*z_l.shape) > self.dropout_rate).astype(float)
                z_l *= mask
                z_l /= (1.0 - self.dropout_rate)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(np.ones_like(z_l))

            z.append(z_l)

        # capa de salida softmax
        W = self.weights[-1]
        b = self.biases[-1]

        a_L = z_l @ W + b
        a.append(a_L)

        z_L = self.softmax(a_L)
        z.append(z_L)

        # z: [(samples, inputs), ..., (samples, neurons_layer_i)]
        # a: [(samples, neurons_layer_i), ...]

        return z, a 

    def loss(self, yhat, ytrue, l2_lambda): # falta ponerle l2 reg
        batch_size = ytrue.shape[0]
        yhat = np.clip(yhat, a_min=1e-9, a_max=1.0) # para que no explote el logaritmo en caso de que yhat=0
        log_probs = -np.sum(ytrue * np.log(yhat), axis=1) # hace la multiplicacion casillero x casillero
        loss = np.sum(log_probs) / batch_size

        l2_reg = l2_lambda * np.sum([np.sum(W**2) for W in self.weights]) / (2 * batch_size)
        loss += l2_reg

        return loss

    def backward(self, activations, pre_activations, ytrue, l2_lambda):
        batch_size = ytrue.shape[0]
        w_grads = [0] * self.nlayers
        b_grads = [0] * self.nlayers

        dz = activations[-1] - ytrue # derivada de softmax + crossentropy

        for layer in reversed(range(self.nlayers)):
            A_prev = activations[layer]
            w_grads[layer] = A_prev.T @ dz / batch_size + (l2_lambda / batch_size) * self.weights[layer]
            b_grads[layer] = np.sum(dz, axis=0, keepdims=True) / batch_size
            if layer > 0:
                dA_prev = dz @ self.weights[layer].T
                dz = dA_prev * self.relu_derivative(pre_activations[layer - 1])

                            # === Aplico la máscara de dropout ===
                if self.dropout_rate > 0:
                    mask = self.dropout_masks[layer - 1]  # porque dropout se aplica solo en ocultas
                    dz *= mask
                    dz /= (1.0 - self.dropout_rate)

        return w_grads, b_grads

    def gradient_descent(self, grads_W, grads_b, lr):
        for i in range(self.nlayers):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

    def adam(self, grads_W, grads_b, lr):

        if not hasattr(self, "m_weights"):
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.t = 1
            self.t += 1
        for i in range(self.nlayers):
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grads_W[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grads_b[i]

            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grads_W[i] ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (grads_b[i] ** 2)

            m_hat_W = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_weights[i] / (1 - self.beta2 ** self.t)

            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)

            self.weights[i] -= lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def lr_scheduling(self, lr_init, current_epoch, total_epochs, type='None', decay_rate=0.995):
        if type == 'None':
            return lr_init
        elif type == 'Linear':
            return (1 - current_epoch / total_epochs) * lr_init
        elif type == 'Exp':
            return lr_init * (decay_rate ** current_epoch)
        else:
            raise ValueError(f'Invalid lr_schedule type: ({type})')
        
    def train(self, X_val, y_val, epochs=50, lr=0.01, batch_size=None, optimizer='GD', early_stopping=False, patience=10, scheduling_type='None', l2_lambda=0, dropout_rate=0, print_progress=True):
        X_train = self.X
        y_train = self.Y

        best_val_loss = float('inf')
        
        if (batch_size is None) or optimizer=='GD':
            batch_size = self.total_samples

        self.dropout_rate = dropout_rate
        self.patience = patience
        self.regularization_lambda = l2_lambda

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            current_lr = self.lr_scheduling(lr, epoch, epochs, type=scheduling_type)

            indices = np.random.permutation(n_samples) # agarro un batch
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            for start in range(0, n_samples, batch_size): # agarro batch por batch
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, pre_acts = self.forward(X_batch, training=True) # PONER DORPOUT OPCIONEAL
                batch_loss = self.loss(activations[-1], y_batch, l2_lambda)
                epoch_loss += batch_loss
                grads_W, grads_b = self.backward(activations, pre_acts, y_batch, l2_lambda)

                if (optimizer == 'GD') or (optimizer == 'SGD'):
                    self.gradient_descent(grads_W, grads_b, current_lr)
                elif optimizer == 'ADAM':
                    self.adam(grads_W, grads_b, current_lr)
                else:
                    raise ValueError(f'Invalid optimizer: ({optimizer})')

            # termine un epoch entero
            number_of_batches = (n_samples / batch_size)
            epoch_loss /= number_of_batches
            self.train_losses.append(epoch_loss)

            val_activations, _ = self.forward(X_val)
            val_loss = self.loss(val_activations[-1], y_val, l2_lambda)
            self.val_losses.append(val_loss)

            if early_stopping:
                if self.patience == 0:
                    if print_progress:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.patience = patience
                    else:
                        self.patience -= 1
            if print_progress:
                if (epoch+1) % 10 == 0 or epoch == epochs - 1:
                    acc = np.mean(self.predict(X_val) == np.argmax(y_val, axis=1))
                    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc * 100:.4f}% - LR: {current_lr:.6f}')
                    
        
    def predict(self, X):
        activations, _ = self.forward(X, training=False)
        prediction = np.argmax(activations[-1], axis=1)
        return prediction

    def accuracy(self, X_train, y_train, X_valid, y_valid, ret=True):
        y_train_pred = self.predict(X_train)
        y_valid_pred = self.predict(X_valid)

        train_acc = np.mean(y_train_pred == np.argmax(y_train, axis=1))
        valid_acc = np.mean(y_valid_pred == np.argmax(y_valid, axis=1))

        if ret:
            print(f'Train Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {valid_acc * 100:.2f}%')
        else:
            return train_acc, valid_acc

    def plot_loss(self):
        print(f'Training Loss: {self.train_losses[-1]:.4f}')
        print(f'Validation Loss: {self.val_losses[-1]:.4f}')
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Cross-Entropy Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, X_train, y_train, X_valid, y_valid):
        y_train_pred = self.predict(X_train)
        y_valid_pred = self.predict(X_valid)

        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        if len(y_valid.shape) > 1:
            y_valid = np.argmax(y_valid, axis=1)

        num_classes = np.max(y_train) + 1 
        cm_train = np.zeros((num_classes, num_classes), dtype=int)
        cm_valid = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred in zip(y_train, y_train_pred):
            cm_train[true, pred] += 1

        for true, pred in zip(y_valid, y_valid_pred):
            cm_valid[true, pred] += 1

        # Plot confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(cm_train, annot=False, fmt='d', ax=axes[0])
        axes[0].set_title('Training Confusion Matrix')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')

        sns.heatmap(cm_valid, annot=False, fmt='d', ax=axes[1])
        axes[1].set_title('Validation Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')

        plt.show()

class NeuralNetworkPytorch(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NeuralNetworkPytorch, self).__init__()
        
        layer_sizes = [input_size] + hidden_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(hidden_layers))
        ])

        self.output = nn.Linear(hidden_layers[-1], num_classes)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.train_losses = []
        self.val_losses = []

    def forward(self, x, ret_softmax=False):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output(x)
        if ret_softmax:
            x = F.softmax(x, dim=1)
        return x
    
    def train_model(self, X_train, y_train, X_val, y_val, batch_size=None, epochs=10):
        if batch_size is None:
            batch_size = X_train.shape[0]

        # proceso los datos de train
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        y_train = torch.argmax(y_train, dim=1)  # paso de matrix one hot a un vector

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # proceso los de val
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        y_val = torch.argmax(y_val, dim=1) # paso de matrix one hot a un vector
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for xbatch, ybatch in dataloader:

                outputs = self.forward(xbatch)
                loss = self.loss_function(outputs, ybatch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            avg_train_loss = total_loss / len(dataloader)
            self.train_losses.append(avg_train_loss)

            self.eval()
            with torch.no_grad(): # freno tempralmente el gradiente
                val_outputs = self.forward(X_val)
                val_loss = self.loss_function(val_outputs, y_val).item()
                self.val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    def accuracy(self, X_train, y_train, X_val, y_val):
        self.eval()

        accuracies = {}
        for X, y, name in [(X_train, y_train, 'Train'), (X_val, y_val, 'Validation')]:
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            y_true = torch.argmax(y, dim=1) if y.ndim > 1 else y.long()

            with torch.no_grad():
                outputs = self.forward(X)
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == y_true).sum().item()
                accuracy = correct / len(y_true)
                accuracies[name] = accuracy
                print(f'{name} Accuracy: {accuracy * 100:.2f}%')

        return accuracies['Train'], accuracies['Validation']
    
    def plot_confusion_matrix(self, X_train, y_train, X_valid, y_valid):
        self.eval()

        # Convert to tensors if needed
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Handle one-hot labels
        y_train_true = torch.argmax(y_train, dim=1) if y_train.ndim > 1 else y_train.long()
        y_valid_true = torch.argmax(y_valid, dim=1) if y_valid.ndim > 1 else y_valid.long()

        with torch.no_grad():
            y_train_pred = torch.argmax(self.forward(X_train), dim=1)
            y_valid_pred = torch.argmax(self.forward(X_valid), dim=1)

        num_classes = max(y_train_true.max(), y_valid_true.max()).item() + 1
        print(num_classes)
        labels = list(range(num_classes))

        cm_train = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        cm_valid = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        for t, p in zip(y_train_true, y_train_pred):
            cm_train[t.item(), p.item()] += 1

        for t, p in zip(y_valid_true, y_valid_pred):
            cm_valid[t.item(), p.item()] += 1

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.heatmap(cm_train.numpy(), annot=False, fmt='d', ax=axes[0])
        axes[0].set_title('Training Confusion Matrix')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')

        sns.heatmap(cm_valid.numpy(), annot=False, fmt='d', ax=axes[1])
        axes[1].set_title('Validation Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')

        plt.tight_layout()
        plt.show()


    def plot_losses(self):
        print(f'Training Loss: {self.train_losses[-1]:.4f}')
        print(f'Validation Loss: {self.val_losses[-1]:.4f}')

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Cross-Entropy Loss Over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.show()

 