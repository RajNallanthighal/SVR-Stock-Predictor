import numpy as np

class SVR:
    """ Defines Support Vector Regression (SVR) model for regression task of predicting a stock's future closing price.
    
    Args:
        learning_rate (float, optional): Controls amount to change the model with respect to the estimated error. Default to 0.003.
        max_iter (int, optional): Controls number of times model iterates over data set. Defaults to 10.
        epsilon (float, optional): Controls margin of tolerance in for which errors are given no penalty. Defaults to 0.004.
        clip_value (float, optional): Controls threshold for outliers in gradients. Defaults to 0.001.
    """
    def __init__(self, learning_rate=0.003, max_iter=10, epsilon=0.004, clip_value=0.001):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.b = 0
        self.w = None
        self.support_vectors = None
        self.support_vector_targets = None
        self.clip_value = clip_value

    def fit(self, X, y):
        """Fits model to data set. Implements epsilon-insensitive loss function in tandem with linear kernel.

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target array.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.support_vectors = np.zeros((0, n_features))
        self.support_vector_targets = np.zeros(0)
        for _ in range(self.max_iter):
            for i in range(n_samples):
                y_pred = self.decision_function(X[i])
                loss = np.maximum(0, np.abs(y_pred - y[i]) - self.epsilon)
                if np.any(loss != 0):
                    update_w = self.learning_rate * ((y[i] - y_pred) * X[i])
                    update_b = self.learning_rate * (y[i] - y_pred)
                    update_w = np.clip(update_w, -self.clip_value, self.clip_value)
                    update_b = np.clip(update_b, -self.clip_value, self.clip_value)
                    self.w += update_w
                    self.b += update_b
                    self.support_vectors = np.vstack([self.support_vectors, X[i]])
                    self.support_vector_targets = np.append(self.support_vector_targets, y[i])

    def decision_function(self, X):
        """Calculates linear kernel given array of input features. 

        Args:
            X (numpy.ndarray): Array of input features.

        Returns:
            Value for input data calculated from linear kernel (dot product).
        """
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """Predicts new target value.

        Args:
            X (numpy.ndarray): Array of input features.

        Returns:
            New target value for data set.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.apply_along_axis(self.decision_function, 1, X)
    
