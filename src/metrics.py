import numpy as np

def accuracy(yhat, ground_truth):
    y_test_indices = np.argmax(ground_truth, axis=1)  # Convert one-hot encoded labels to class indices
    test_acc = np.mean(yhat == y_test_indices)  # Compare predicted and true class indices
    print("Accuracy:", test_acc)

# PARA
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