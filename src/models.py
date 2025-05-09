import numpy as np
import matplotlib.pyplot as plt

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

        self.patience = 10

        np.random.seed(42)

        for i in range(self.nlayers):
            std = np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.normal(0, std, size=(layer_sizes[i], layer_sizes[i+1])) # inicializo pesos con glorot
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(W)
            self.biases.append(b)

        # inicializo momentos para adam

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

        for i in range(self.nlayers - 1): # capas ocultas relu
            W = self.weights[i]
            b = self.biases[i]

            a_l = z[-1] @ W + b
            a.append(a_l)

            z_l = self.relu(a_l)
            z.append(z_l)

            z_l = self.dropout(z_l, dropout_rate=0.15, training=training)
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
        log_probs = -np.sum(ytrue * np.log(yhat), axis=1) # hace la multiplicacion casillero x casillero
        loss = np.sum(log_probs) / batch_size

        l2_reg = l2_lambda * np.sum([np.sum(W**2) for W in self.weights]) / (2 * batch_size)
        loss += l2_reg

        return loss

    def backward(self, activations, pre_activations, ytrue, l2_lambda):
        batch_size = ytrue.shape[0]
        w_grads = [0] * self.nlayers
        b_grads = [0] * self.nlayers

        dZ = activations[-1] - ytrue # derivada de softmax + crossentropy

        for l in reversed(range(self.nlayers)):
            A_prev = activations[l]
            w_grads[l] = A_prev.T @ dZ / batch_size + (l2_lambda / batch_size) * self.weights[l]
            b_grads[l] = np.sum(dZ, axis=0, keepdims=True) / batch_size
            if l > 0:
                dA_prev = dZ @ self.weights[l].T
                dZ = dA_prev * self.relu_derivative(pre_activations[l - 1])

        return w_grads, b_grads

    def gradient_descent(self, grads_W, grads_b, lr):
        for i in range(self.nlayers):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

    def adam(self, grads_W, grads_b, lr, beta1=0.9, beta2=0.999, epsilon=1e-8): # cambiar l por i en cada metodo
        return

    def dropout(self, z_list, dropout_rate=0.15, training=True):
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError('Invalid dropout_rate. Must be in the range [0, 1).')
        
        if not training:
            return z_list
        
        mask = np.random.rand(*z_list.shape) > dropout_rate

        z_l = z_l * mask
        z_l /= (1 - dropout_rate)

        return z_l

    def lr_scheduling(self, lr_init, current_epoch, total_epochs, type='None', decay_rate=0.995):
        if type == 'None':
            return lr_init
        elif type == 'Linear':
            return (1 - current_epoch / total_epochs) * lr_init
        elif type == 'Exp':
            return lr_init * (decay_rate ** current_epoch)
        else:
            raise ValueError(f'Invalid lr_schedule type: ({type})')
        
    def train(self, X_val, y_val, epochs=50, lr=0, batch_size=None, optimizer='gradient_descent', early_stopping=False, scheduling_type='None', l2_lambda=0.1): # cambiar a total_samples
        X_train = self.X
        y_train = self.Y

        best_val_loss = float('inf')
        
        if batch_size is None:
            batch_size = self.total_samples

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle the data
            current_lr = self.lr_scheduling(lr, epoch, epochs, type=scheduling_type)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, pre_acts = self.forward(X_batch, training=True) # PONER DORPOUT OPCIONEAL
                batch_loss = self.loss(activations[-1], y_batch, l2_lambda)
                epoch_loss += batch_loss
                grads_W, grads_b = self.backward(activations, pre_acts, y_batch, l2_lambda)

                if optimizer == 'gradient_descent':
                    self.gradient_descent(grads_W, grads_b, current_lr)
                elif optimizer == 'ADAM':
                    self.adam(grads_W, grads_b, current_lr)
                else:
                    raise ValueError(f'Invalid optimizer: ({type})')

            # termine un epoch entero
            epoch_loss /= (n_samples / batch_size)
            self.train_losses.append(epoch_loss)

            val_activations, _ = self.forward(X_val)
            val_loss = self.loss(val_activations[-1], y_val, l2_lambda)
            self.val_losses.append(val_loss)

            if early_stopping:
                if self.patience == 0:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.patience = 10
                    else:
                        self.patience -= 1

            if (epoch+1) % 10 == 0 or epoch == epochs - 1:
                acc = np.mean(self.predict(X_val) == np.argmax(y_val, axis=1))
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f} - LR: {current_lr:.6f}')
    
    def predict(self, X):
        activations, _ = self.forward(X, training=False)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X_train, y_train, X_valid, y_valid):
        y_train_pred = self.predict(X_train)
        y_valid_pred = self.predict(X_valid)

        train_acc = np.mean(y_train_pred == np.argmax(y_train, axis=1))
        valid_acc = np.mean(y_valid_pred == np.argmax(y_valid, axis=1))

        print(f'Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}')

    def plot_loss(self):
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
