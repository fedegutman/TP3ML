import numpy as np

class NeuralNetwork:
    def __init__(self, X:np.ndarray, y:np.ndarray, layer_sizes:list):
        self.X = X # (samples, features)
        self.Y = y # (samples, classes)
        self.total_samples = y.shape[0]

        self.nlayers = len(layer_sizes) - 1 # sin contar el input
        self.weights = []
        self.biases = []

        np.random.seed(42)

        for i in range(self.nlayers):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.uniform(-limit, limit, size=(layer_sizes[i], layer_sizes[i+1])) # glorot CHECK
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(W)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True)) # keepdims: (n,) -> (n,1)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        z = [X]
        a = []

        for i in range(self.nlayers - 1): # capas ocultas relu
            W = self.weights[i]
            b = self.biases[i]

            a_l = z[-1] @ W + b
            a.append(a_l)

            z_l = self.relu(a_l)
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

    def compute_loss(self, Y_hat, Y_true):
        batch_size = Y_true.shape[0]
        # log_probs = -np.log(Y_hat[np.arange(batch_size), Y_true]) # chequear
        log_probs = -np.sum(Y_true * np.log(Y_hat), axis=1)  # esto si ya encode
        loss = np.sum(log_probs) / batch_size

        return loss

    def backward(self, activations, pre_activations, Y_true):
        batch_size = Y_true.shape[0]
        w_grads = [0] * self.nlayers
        b_grads = [0] * self.nlayers

        dZ = activations[-1] - Y_true # derivada de softmax + crossentropy

        for l in reversed(range(self.nlayers)):
            A_prev = activations[l]
            w_grads[l] = A_prev.T @ dZ / batch_size
            b_grads[l] = np.sum(dZ, axis=0, keepdims=True) / batch_size
            if l > 0:
                dA_prev = dZ @ self.weights[l].T
                dZ = dA_prev * self.relu_derivative(pre_activations[l - 1])

        return w_grads, b_grads

    def gradient_descent(self, grads_W, grads_b, lr):
        for l in range(self.nlayers):
            self.weights[l] -= lr * grads_W[l]
            self.biases[l] -= lr * grads_b[l]

    def adam(self, grads_W, grads_b, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        return


    def lr_scheduling(self, lr_init, current_epoch, total_epochs, type='None', decay_rate=0.96, decay_steps=500):
        if type == 'None':
            return lr_init
        elif type == 'Linear':
            return (1 - current_epoch / total_epochs) * lr_init
        elif type == 'Exp':
            return lr_init * (decay_rate ** (current_epoch/decay_steps))
        else:
            raise ValueError(f'Invalid lr_schedule type: ({type})')
        
    def train(self, X_val, y_val, epochs=50, lr=0.01, batch_size=None, optimizer='gradient_descent', scheduling_type='None'): # cambiar a total_samples
        X_train = self.X
        y_train = self.Y
        
        if batch_size is None:
            batch_size = self.total_samples

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle the data
            current_lr = self.lr_scheduling(lr, epoch, epochs, type=scheduling_type)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, pre_acts = self.forward(X_batch)
                loss = self.compute_loss(activations[-1], y_batch)
                grads_W, grads_b = self.backward(activations, pre_acts, y_batch)
                if optimizer == 'gradient_descent':
                    self.gradient_descent(grads_W, grads_b, current_lr)
                elif optimizer == 'ADAM':
                    self.adam(grads_W, grads_b, current_lr)
                else:
                    raise ValueError(f'Invalid optimizer: ({type})')

            # performance en validation dsp de un epoch entero
            val_preds = self.predict(X_val)
            y_val_indices = np.argmax(y_val, axis=1)  # Convert one-hot encoded labels to class indices
            acc = np.mean(val_preds == y_val_indices)  # Compare predicted and true class indices
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Val Acc: {acc:.4f} - LR: {current_lr:.6f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
