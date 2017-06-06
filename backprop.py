import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

# 400,3
n_records, n_features = features.shape
last_loss = None
# Initialize weights
# 3:2
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
# 1:2
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    # 3:2
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    # 1:2
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets): # 400:3 400:1
        ## Forward pass ##
        # TODO: Calculate the output
        # 400:2
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output,weights_hidden_output[:,None])

        ## Backward pass ##
        # TODO: Calculate the error
        # 400:1
        error = y - output

        # TODO: Calculate error gradient in output unit
        # 400:1
        output_error = error * output * (1 - output)

        # TODO: propagate errors to hidden layer
        # weights_hidden_output * del_err_output * hidden_layer_output * (1-hidden_layer_output)
        # 1:2 400:1 400:2 = 400:1
        hidden_error = weights_hidden_output * output_error * hidden_output * (1-hidden_output)

        # TODO: Update the change in weights
        # 400:1 , 400:2 = 1:2
        del_w_hidden_output += np.dot(output_error.T, hidden_output)
        # 400:1 , 400:3 = 1:3
        del_w_input_hidden += np.dot(hidden_error.T, x)

    # TODO: Update weights
    weights_input_hidden += learnrate * x * del_w_input_hidden.T
    weights_hidden_output += learnrate * hidden_output * del_w_hidden_output.T

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
