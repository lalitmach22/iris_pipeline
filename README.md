 
# Iris Classification Pipeline 🧠🌸

This project demonstrates a complete CI/CD pipeline for a machine learning project using GitHub Actions and Continuous Machine Learning (CML). It includes model training, testing, linting, formatting checks, performance visualization, and automatic PR commenting.

---

## 🔧 Features Implemented

### ✅ CI/CD Automation via GitHub Actions
- Every PR to `main` triggers a GitHub Actions workflow:
  - Sets up Python environment
  - Installs dependencies
  - Runs training (`src/train.py`)
  - Runs tests using `pytest`
  - Auto-formats with `black`
  - Lints code with `flake8`
  - Generates model performance metrics (`src/plot_metrics.py`)
  - Comments results directly on the PR using CML
  - Uploads artifacts (`model.pkl`, `metrics.png`, `report.md`) for review

### 📦 Artifacts
Uploaded artifacts per run include:
- `model.pkl`: Trained model
- `metrics.png`: Confusion matrix and metrics
- `report.md`: Full CI/CD report with lint, test, and performance info

### 🧪 Model Evaluation
- Accuracy, precision, recall, and F1-score calculated on test data
- Visual confusion matrix plotted and included in PR
- Report posted as a PR comment using `cml comment`

---

## 📂 Project Structure

```
iris_pipeline/
├── src/
│   ├── train.py              # Trains and saves the model
│   ├── plot_metrics.py       # Evaluates and visualizes performance
├── tests/
│   └── test_model.py         # Unit tests
├── .github/
│   └── workflows/
│       └── sanity-test.yml   # GitHub Actions workflow
├── requirements.txt
└── README.md
```

---

## 🛠️ Commands to Run Locally

```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Generate metrics
python src/plot_metrics.py

# Run tests
pytest

# Format and lint
black src/ tests/
flake8 src/ tests/
```

---

## 🤖 GitHub Actions Workflow

```yaml
on:
  pull_request:
    branches: [main]

permissions:
  pull-requests: write
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    ...
```

---

## ✅ Example PR Comment from CML

```
## Code Formatting (Black)
All files are properly formatted ✅

## Linting Results (Flake8)
No linting issues ❌

## Test Results
2 passed in 3.05s ✅

## Model Performance Metrics
[Confusion matrix image shown here]
```

---

## 📈 Result
Fully automated CI/CD for ML model sanity checking and performance feedback — **without needing to open the terminal manually during PRs!** 🎯

