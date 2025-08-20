
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any
import os

class ModelEvaluator:
    def __init__(self, save_plots: bool = True):
        self.save_plots = save_plots
        
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        if self.save_plots:
            self._plot_confusion_matrix(results['confusion_matrix'], model.classes_)
            
        return results
    
    def analyze_weights(self, model) -> Dict[str, Any]:
        """Analyze model weights"""
        if hasattr(model, 'get_weights'):
            weights_info = model.get_weights()
        elif hasattr(model, 'coef_'):
            weights_info = {
                'coef': model.coef_,
                'intercept': getattr(model, 'intercept_', None)
            }
        else:
            return {}
        
        analysis = {}
        
        if 'coef' in weights_info:
            coef = weights_info['coef']
            analysis.update({
                'weight_norm': np.linalg.norm(coef),
                'weight_sparsity': np.mean(np.abs(coef) < 1e-6),
                'weight_std': np.std(coef),
                'weight_mean': np.mean(coef),
                'max_weight': np.max(np.abs(coef))
            })
            
            if self.save_plots:
                self._plot_weight_distribution(coef)
                
        return analysis
    
    def _plot_confusion_matrix(self, cm: np.ndarray, classes: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_weight_distribution(self, weights: np.ndarray):
        """Plot weight distribution"""
        plt.figure(figsize=(12, 4))
        
        # Weight histogram
        plt.subplot(1, 3, 1)
        plt.hist(weights.flatten(), bins=50, alpha=0.7)
        plt.title('Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        
        # Weight heatmap (if multiple classes)
        if weights.ndim > 1:
            plt.subplot(1, 3, 2)
            sns.heatmap(weights, cmap='RdBu_r', center=0, cbar=True)
            plt.title('Weight Matrix')
            plt.xlabel('Feature')
            plt.ylabel('Class')
        
        # Weight magnitude per class
        plt.subplot(1, 3, 3)
        if weights.ndim > 1:
            weight_norms = np.linalg.norm(weights, axis=1)
            plt.bar(range(len(weight_norms)), weight_norms)
            plt.title('Weight Norm per Class')
            plt.xlabel('Class')
            plt.ylabel('L2 Norm')
        else:
            plt.bar(range(len(weights)), np.abs(weights))
            plt.title('Absolute Weights')
            plt.xlabel('Feature')
            plt.ylabel('|Weight|')
        
        plt.tight_layout()
        plt.savefig('weight_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()