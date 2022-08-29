import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self, iterations=60000, learning_step_size=0.1, verbose=True, logging_intervals=10):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.iterations = iterations
        self.learning_step_size = learning_step_size
        self.verbose = verbose
        self.logging_intervals = logging_intervals
        self.theta = None
        self.bias = 0

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels
        """

        # Transform input to numpy array, and stack a bias column onto it
        X_np = np.array(X)
        samples = len(X_np[0])
        features = len(X_np[1])
        self.theta = np.zeros(samples)

        for i in range(self.iterations):

            if self.verbose and i % (round(self.iterations/self.logging_intervals)) == 0:
                print(f"Iteration {i}/{self.iterations}")

            # Prediction
            linear_model = np.dot(X_np, self.theta) + self.bias
            prediction = sigmoid(linear_model)

            # Gradient Descent
            error = prediction - y
            delta_theta = (1/features)*np.dot(error, X_np)
            delta_bias = (1/features)*np.sum(error)
            self.theta = self.theta - self.learning_step_size*delta_theta
            self.bias = self.bias - self.learning_step_size*delta_bias

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        return np.array(sigmoid(np.dot(X, self.theta) + self.bias))

# --- Some utility functions


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input
    """
    return 1. / (1. + np.exp(-x))
