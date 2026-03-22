# MLOps Demo Project

A complete, reproducible Machine Learning Operations (MLOps) pipeline demonstrating best practices for training, evaluating, and deploying ML models in production.

Perfect for conferences, workshops, and educational purposes!

## 📋 Overview

This project showcases a complete MLOps workflow with:

- **Data Pipeline**: Load, preprocess, and validate data
- **Model Training**: Train and evaluate ML models
- **Experiment Tracking**: Store metrics and model artifacts
- **CI/CD Automation**: Automated testing and retraining via GitHub Actions
- **Testing**: Unit tests for data and model pipelines
- **Best Practices**: Configuration management, logging, and reproducibility

## 📁 Project Structure

```
├── src/                      # Source code
│   ├── data/                 # Data loading & preprocessing
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── models/               # Model training & evaluation
│   │   ├── model.py
│   │   └── __init__.py
│   └── pipeline/             # Main ML pipeline orchestration
│       ├── pipeline.py
│       └── __init__.py
├── config/
│   └── config.yaml           # Configuration file
├── data/                     # Input data (generated)
├── models/                   # Trained models & artifacts
│   └── metrics/              # Evaluation metrics
├── tests/                    # Unit tests
│   └── test_pipeline.py
├── .github/workflows/        # CI/CD workflows
│   └── train.yml             # Automated training pipeline
├── setup.sh                  # One-click setup script
├── train.py                  # Entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd mlops-demo

# Run one-click setup (Linux/Mac)
bash setup.sh

# Or on Windows PowerShell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
python train.py
```

### Option 2: Step-by-Step Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd mlops-demo

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest tests/ -v

# 5. Run the ML pipeline
python train.py
```

## 📊 Pipeline Workflow

The ML pipeline executes the following steps:

1. **Load Data**: Uses the Iris dataset (standard ML benchmark)
2. **Preprocessing**: Normalize features using StandardScaler
3. **Train Model**: Train Random Forest classifier
4. **Evaluate**: Calculate accuracy, precision, recall, F1-score
5. **Save Artifacts**: Store model, scaler, and metrics

### Output Files

After running the pipeline, check:

- `models/model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/metrics/metrics.json` - Evaluation metrics
- `data/*.npy` - Dataset splits

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

## 📝 Configuration

Edit `config/config.yaml` to modify:

- Model hyperparameters
- Data split ratios
- Random seeds (for reproducibility)
- Output paths

Example:
```yaml
data:
  test_size: 0.2
  random_state: 42

model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
```

## 🎓 Learning Outcomes

After running this project, you'll understand:

- ✅ How to structure ML code professionally
- ✅ Data preprocessing and feature scaling
- ✅ Model training and evaluation best practices
- ✅ Storing and versioning models
- ✅ Automated testing for ML pipelines
- ✅ CI/CD automation for model retraining
- ✅ Configuration management
- ✅ Reproducibility and experiment tracking

## 🔧 Extending the Project

### Add a New Model

Edit `src/models/model.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

def train_model(X_train, y_train, random_state=42):
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    return model
```

### Use a Different Dataset

Edit `src/data/data_loader.py`:

```python
from sklearn.datasets import load_wine

iris = load_wine()  # Or any other dataset
X = iris.data
y = iris.target
```

### Add Model Versioning

Install MLflow and track experiments:

```bash
pip install mlflow
```

Then log metrics:

```python
import mlflow
mlflow.log_params({"n_estimators": 100})
mlflow.log_metrics(metrics)
mlflow.sklearn.log_model(model, "model")
```

### Deploy with FastAPI

Create `app.py`:

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(features):
    return {"prediction": model.predict([features])[0]}
```

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/)
- [DVC (Data Version Control)](https://dvc.org/)

## ⚠️ System Requirements

- Python 3.8+
- pip or conda
- ~500 MB disk space

## 📄 License

MIT License - Feel free to use and modify for educational and commercial purposes.

## 🤝 Contributing

Have improvements? Fork and submit a pull request!

## ❓ Troubleshooting

### ImportError: No module named 'src'

Ensure you're running from the project root directory and have installed dependencies.

### Permission denied: ./setup.sh

On Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

### Python version mismatch

Ensure Python 3.8+ is installed:
```bash
python3 --version
```

---

**Happy Learning! 🚀**

Questions? Check the source code comments or open an issue.
