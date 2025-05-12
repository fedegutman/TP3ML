import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing import one_hot_encoding
from src.models import NeuralNetwork, NeuralNetworkPytorch
import torch
'''
import pandas as pd

# Example values (replace these with your actual computed metrics)
train_loss = 0.25
train_acc = 0.92
valid_loss = 0.30
valid_acc = 0.89

# Create a DataFrame for the table
metrics_table = pd.DataFrame({
    "Set": ["Training", "Validation"],
    "Accuracy": [train_acc, valid_acc],
    "Loss": [train_loss, valid_loss]
})

# Display the table
print(metrics_table)
'''

def plot_model_performance(models:list, X_train, y_train, X_test, y_test): # ESTO FUNCIONA SOLO PARA LOS MODELOS DE NN NO LOS DE PYTORCH
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    model_names = [f'Model {i}' for i in range(len(models))]

    train_accuracies = []
    test_accuracies = []

    train_loss = []
    test_loss = []

    for model in models:
        if isinstance(model, NeuralNetwork):
            train_acc, test_acc = model.accuracy(X_train, y_train, X_test, y_test, ret=False)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            train_loss.append(model.train_losses[-1])
            activations, _= model.forward(X_test, training=False)
            yhat = activations[-1]
            loss = model.loss(yhat, y_test, model.regularization_lambda)
            test_loss.append(loss)

        # elif model is NeuralNetworkPytorch:
        else:
            train_acc, test_acc = model.accuracy(X_train, y_train, X_test, y_test)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            train_loss.append(model.train_losses[-1])
            yhat = model.forward(torch.tensor(X_test, dtype=torch.float32))
            loss = model.loss_function(yhat, torch.tensor(y_test, dtype=torch.float32))
            test_loss.append(loss.detach().numpy())

    x = np.arange(len(models))
    w = 0.35

    # acc
    axes[0].bar(x - w/2, train_accuracies, width=w, label='Train Accuracy')
    axes[0].bar(x + w/2, test_accuracies, width=w, label='Test Accuracy')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()

    # loss
    axes[1].bar(x - w/2, train_loss, width=w, label='Train Loss')
    axes[1].bar(x + w/2, test_loss, width=w, label='Test Loss')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    
    plt.show()
        

