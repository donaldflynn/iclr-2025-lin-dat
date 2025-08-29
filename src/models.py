
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import Dict, Any
from omegaconf import DictConfig

class LinearMulticlass(BaseEstimator, ClassifierMixin):
    """
    Custom linear regression approach:
    1. Center data (mean = 0) - handled in preprocessing
    2. Convert labels to +1/-1 for each class  
    3. Train binary linear regression for each class
    """
    
    def __init__(self, regularization: float = 0.01, fit_intercept: bool = False):
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit one binary linear regression per class"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_data, n_features = X.shape
        
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        
        for i, class_label in enumerate(self.classes_):
            # Convert to binary labels: +1 for this class, -1 for others
            y_binary = np.where(y == class_label, 1, -1)
            
            # Solve linear regression: minimize ||Xw - y||^2 + λ||w||^2
            if self.regularization > 0:
                # Ridgae regression solution: w = (X'X + λI)^-1 X'y
                XtX = 1/n_data * (X.T @ X)
                regularization_matrix = self.regularization * np.eye(n_features)
                w = np.linalg.solve((XtX) + regularization_matrix, 1/n_data * X.T @ y_binary)
            else:
                # Ordinary least squares: w = (X'X)^-1 X'y
                w = np.linalg.lstsq(X, y_binary, rcond=None)[0]
            
            self.coef_[i] = w
            
            if self.fit_intercept:
                self.intercept_[i] = np.mean(y_binary - X @ w)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by choosing class with highest linear output"""
        scores = X @ self.coef_.T + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return decision scores for each class"""
        return X @ self.coef_.T + self.intercept_
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return weights for analysis"""
        return {
            'coef': self.coef_,
            'intercept': self.intercept_,
            'classes': self.classes_
        }

def create_model(cfg: DictConfig):
    """Create model based on configuration"""
    if cfg.model.type == "logistic":
        return LogisticRegression(
            C=1.0/cfg.model.regularization,
            max_iter=cfg.model.max_iter,
            solver=cfg.model.solver,
            multi_class=cfg.model.multi_class,
            random_state=cfg.seed
        )
    elif cfg.model.type == "linear_multiclass":
        return LinearMulticlass(
            regularization=cfg.model.regularization,
            fit_intercept=cfg.model.get('fit_intercept', False)
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
