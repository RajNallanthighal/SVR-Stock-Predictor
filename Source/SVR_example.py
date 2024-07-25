import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

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
    
def is_market_open(date):
    """Helper function to check if a date is a valid NYSE trading day.
   
    Args:
        date (datetime.date): A specific date.

    Returns:
        bool: True if 'date' is a valid trading day, False otherwise.
    """
    holidays = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 15),
        datetime(2024, 2, 19),
        datetime(2024, 3, 29),
        datetime(2024, 5, 27),
        datetime(2024, 6, 19),
        datetime(2024, 7, 4),
        datetime(2024, 9, 2),
        datetime(2024, 11, 28),
        datetime(2024, 12, 25),
        datetime(2025, 1, 1),
        datetime(2025, 1, 20),
        datetime(2025, 2, 17),
        datetime(2025, 4, 18),
        datetime(2025, 5, 26),
        datetime(2025, 6, 19),
        datetime(2025, 7, 4),
        datetime(2025, 9, 1),
        datetime(2025, 11, 27),
        datetime(2025, 12, 25),
        datetime(2026, 1, 1),
        datetime(2026, 1, 19),
        datetime(2026, 2, 16),
        datetime(2026, 4, 3),
        datetime(2026, 5, 25),
        datetime(2026, 6, 19),
        datetime(2026, 7, 3),
        datetime(2026, 9, 7),
        datetime(2026, 11, 26),
        datetime(2026, 12, 25)
        # (...) add additional holidays as needed
    ]
    return date.weekday() < 5 and date not in holidays

def get_next_trading_day(start_date):
    """Calculates the next valid trading day from a specific date.

    Args:
        start_date (datetime.date): A specific date.

    Returns:
        The next valid trading day.
    """
    next_day = start_date + timedelta(days=1)
    while not is_market_open(next_day):
        next_day += timedelta(days=1)
    return next_day

if __name__ == "__main__":
    """Handles core program functionality for getting S&P500 stock data, creating/fitting model, 
    and printing next trading day close price prediction.
    """
    # Getting stock data.
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.index = pd.to_datetime(sp500.index)
    # Preparing input features data set.
    sp500 = sp500.loc["2000-01-01":].copy()
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    target = sp500["Tomorrow"].to_numpy()
    target = target[:-1] # truncates Nan value
    features = sp500[["Close", "Open", "High", "Low", "Volume"]].to_numpy()
    input_features = features[-1]
    features = features[:-1]
    # Scaling features and target data sets.
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    target = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    features = feature_scaler.fit_transform(features)
    input_features = feature_scaler.transform(input_features.reshape(1, -1)).flatten()
    # Creating and fitting SVR model.
    svr = SVR()
    svr.fit(features, target)
    scaled_prediction = svr.decision_function(input_features.reshape(1, -1))
    unscaled_prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).flatten()
    # Printing prediction for next trading day's close price.
    print(f"{get_next_trading_day(datetime.today().date())} Close Prediction: ${unscaled_prediction[0]:,.2f}")
    