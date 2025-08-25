
import hydra
from omegaconf import DictConfig
import numpy as np
import pickle
import logging
from src.data import DataLoader
from src.models import create_model
from src.evaluation import ModelEvaluator

# Set up logging
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Set random seed for reproducibility
    np.random.seed(cfg.seed)
    
    log.info(f"Starting experiment: {cfg.experiment_name}")
    log.info(f"Dataset: {cfg.dataset.name}, Model: {cfg.model.type}")
    
    # Load data
    data_loader = DataLoader(cfg)
    X_train, y_train, X_test_pois, y_test_pois, X_test_clean, y_test_clean = data_loader.load_data()
    
    log.info(f"Data loaded - Train: {X_train.shape}, Test_clean: {X_test_clean.shape}, Test_pois: {X_test_pois.shape}")
    
    # Create and train model
    model = create_model(cfg)
    log.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator(save_plots=cfg.save_plots)
    
    # Performance evaluation
    results = evaluator.evaluate(model, X_test_pois, y_test_pois, X_test_clean, y_test_clean)
    log.info(f"Clean Accuracy: {results['clean_accuracy']:.4f}")
    log.info(f"Poison Accuracy: {results['poison_accuracy']:.4f}")


    # Weight analysis
    weight_analysis = evaluator.analyze_weights(model)
    log.info(f"Weight Analysis: {weight_analysis}")
    
    # Save model and results
    if cfg.save_model:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        log.info("Model saved")
    
    # Save detailed results
    all_results = {
        'config': dict(cfg),
        'performance': results,
        'weight_analysis': weight_analysis,
        'data_shapes': {
            'train': X_train.shape,
            'test_clean': X_test_clean.shape,
            'test_poison': X_test_pois.shape
        }
    }
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    log.info("Experiment completed!")
    
    # return results['accuracy']  # Return metric for hyperparameter optimization

if __name__ == "__main__":
    train()