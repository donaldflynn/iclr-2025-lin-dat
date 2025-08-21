# ML Experiments

Machine learning experiment framework using Hydra for configuration management.

## Setup

```bash
pip install requirements.txt
```

## Usage

### Basic Commands

```bash
# Default: CIFAR-10 with logistic regression
python train.py

# Different datasets
python train.py dataset=mnist
python train.py dataset=synthetic

# Different models
python train.py model=linear_multiclass

# Override parameters
python train.py model.regularization=0.1 seed=123

# Center data (set mean to 0)
python train.py dataset.center_data=true
```

### Multiple Experiments

```bash
# Run multiple seeds
python train.py -m seed=42,123,456

# Test different regularization values
python train.py -m model.regularization=0.001,0.01,0.1,1.0

# Compare models on same dataset
python train.py -m model=logistic,linear_multiclass

# Test noise levels
python train.py -m dataset.poison.add_noise=0.0,0.1,0.2
```

## Project Structure

```
ml_experiments/
├── conf/
│   ├── config.yaml              # Main configuration
│   ├── dataset/                 # Dataset configs
│   │   ├── cifar10.yaml
│   │   ├── mnist.yaml
│   │   └── synthetic.yaml
│   └── model/                   # Model configs
│       ├── logistic.yaml
│       └── linear_multiclass.yaml
├── src/
│   ├── data.py                  # Data loading
│   ├── models.py               # Model implementations
│   └── evaluation.py           # Evaluation code
├── train.py                     # Main script
├── data/                        # Downloaded datasets
└── outputs/                     # Results
    └── experiment_name/
        └── 2024-01-15_14-30-45/
            ├── .hydra/config.yaml    # Config used
            ├── model.pkl             # Trained model
            ├── results.pkl           # All results
            ├── confusion_matrix.png
            ├── weight_analysis.png
            └── train.log
```

## Configuration

### Available Datasets
- `cifar10`: CIFAR-10 (10 classes, 32x32x3 images)
- `mnist`: MNIST (10 classes, 28x28 grayscale)
- `synthetic`: Generated Gaussian data

### Available Models
- `logistic`: Scikit-learn LogisticRegression
- `linear_multiclass`: Custom linear regression per class

### Common Overrides

```bash
# Model parameters
python train.py model.regularization=0.01
python train.py model.max_iter=2000

# Data preprocessing
python train.py dataset.normalize=true
python train.py dataset.center_data=true
python train.py dataset.flatten=true

# Data poison
python train.py dataset.poison.add_noise=0.1

# Experiment settings
python train.py experiment_name="my_test"
python train.py seed=999
```

## Results

Each experiment creates a timestamped directory with:
- `model.pkl`: Trained model
- `results.pkl`: Performance metrics and analysis
- `confusion_matrix.png`: Confusion matrix plot
- `weight_analysis.png`: Weight distribution plots
- `.hydra/config.yaml`: Exact configuration used

### Loading Results

```python
import pickle

# Load results
with open('outputs/baseline/2024-01-15_14-30-45/results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"Accuracy: {results['performance']['accuracy']:.3f}")
print(f"Config used: {results['config']}")
```

### Comparing Experiments

```python
import pickle
import pandas as pd

# Load multiple experiments
experiments = [
    'outputs/baseline/2024-01-15_14-30-45',
    'outputs/baseline/2024-01-15_14-45-30'
]

data = []
for exp_dir in experiments:
    with open(f'{exp_dir}/results.pkl', 'rb') as f:
        r = pickle.load(f)
    data.append({
        'model': r['config']['model']['type'],
        'accuracy': r['performance']['accuracy'],
        'regularization': r['config']['model']['regularization']
    })

df = pd.DataFrame(data)
print(df)
```

## Models

### Logistic Regression
Standard scikit-learn LogisticRegression with configurable regularization.

### Linear Multiclass
Custom implementation that:
1. Centers data (mean = 0)
2. Converts labels to +1/-1 for each class
3. Trains binary linear regression for each class

## Custom Experiments

Create `conf/experiment/my_experiment.yaml`:

```yaml
# @package _global_
defaults:
  - override /dataset: mnist
  - override /model: linear_multiclass

experiment_name: "mnist_linear_test"

model:
  regularization: 0.001
dataset:
  center_data: true

seed: 42
```

Run with:
```bash
python train.py experiment=my_experiment
```

## Adding Models

1. Implement in `src/models.py`:
```python
class NewModel(BaseEstimator, ClassifierMixin):
    def __init__(self, param=1.0):
        self.param = param
    
    def fit(self, X, y):
        # Implementation
        return self
    
    def predict(self, X):
        # Implementation
        return predictions
```

2. Add to `create_model()` function in `src/models.py`

3. Create config file `conf/model/new_model.yaml`

## Troubleshooting

**Module import errors**:
- Run from repository root directory

**Data preprocessing issues**:
- Check logs for preprocessing steps and value ranges