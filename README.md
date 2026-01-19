# Fast Causal SHAP
[![PyPI version](https://badge.fury.io/py/fast-causal-shap.svg)](https://badge.fury.io/py/fast-causal-shap)
[![Docs](https://github.com/woonyee28/FastCausalSHAP/actions/workflows/pages/pages-build-deployment/badge.svg)](https://woonyee28.github.io/FastCausalSHAP/)&nbsp;&nbsp;
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/woonyee28/FastCausalSHAP/actions/workflows/test-python-versions.yml/badge.svg)](https://github.com/woonyee28/FastCausalSHAP/actions/workflows/test-python-versions.yml)

Fast Causal SHAP is a Python package designed for efficient and interpretable SHAP value computation in causal inference tasks. It integrates seamlessly with various causal inference frameworks and enables feature attribution with awareness of causal dependencies.

## Features

- Fast computation of SHAP values for causal models
- Support for multiple causal inference frameworks

## Installation

Install the stable version via PyPI:

```bash
pip install fast-causal-shap
```

For Development
```bash
  Clone and install in editable mode with development dependencies:
  git clone https://github.com/woonyee28/CausalSHAP.git
  cd CausalSHAP
  pip install -e ".[dev]"
  pre-commit install
```

## Quick Start
```python
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from fast_causal_shap import FastCausalSHAP

# Load your data
data = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 4, 6, 8, 10],
    'Y': [3, 6, 9, 12, 15]
})

# Train a model
model = RandomForestRegressor()
X = data[['X1', 'X2']]
y = data['Y']
model.fit(X, y)

# Causal effects JSON file
# This defines the causal relationships: X1 -> Y and X2 -> Y
causal_effects = [
    {"Pair": "X1->Y", "Mean_Causal_Effect": 0.8},
    {"Pair": "X2->Y", "Mean_Causal_Effect": 0.5}
]
with open('causal_effects.json', 'w') as f:
    json.dump(causal_effects, f)

# Initialize FastCausalSHAP
shap_explainer = FastCausalSHAP(data, model, target_variable='Y')

# Load causal graph
shap_explainer.load_causal_strengths('causal_effects.json')

# Compute SHAP values for a single instance
shap_values = shap_explainer.compute_modified_shap_proba(data.iloc[0])
print(shap_values)
```

### Working with Different Causal Inference Algorithms

Fast Causal SHAP supports integration with structural algorithms such as:
1. Peter-Clarke (PC) Algorithm
2. IDA Algorithm
3. Fast Causal Inference (FCI) Algorithm
You can find example R code for these integrations here: [FastCausalSHAP R code examples](https://github.com/woonyee28/CausalSHAP/tree/main/code/r)

Generate your causal graph using your preferred algorithm, then export to JSON format.

Format of the Causal_Effect.json:
```
[
  {
    "Pair": "Bacteroidia->Clostridia",
    "Mean_Causal_Effect": 0.71292
  },
  {
    "Pair": "Clostridia->Alphaproteobacteria",
    "Mean_Causal_Effect": 0.37652
  }, ......
]
```

## Development and Contributions
Setup Development Environment
```bash
  # Clone repository
  git clone https://github.com/woonyee28/CausalSHAP.git
  cd CausalSHAP

  # Install with dev dependencies
  pip install -e ".[dev]"

  # Install pre-commit hooks
  pre-commit install

  # Running Tests
  pre-commit run --all-files

  # Run all tests
  pytest

  # Run with coverage
  pytest --cov=fast_causal_shap --cov-report=html

  # View coverage report
  open htmlcov/index.html
```
Code Quality

This project uses automated code quality tools:
  - Black: Code formatting
  - isort: Import sorting
  - Flake8: Linting
  - mypy: Type checking


##  Troubleshooting / FAQ

**Q: I get "Must call load_causal_strengths() before computing SHAP values"**
- A: You need to load a causal graph before computing SHAP values. Call `load_causal_strengths()` first.

**Q: "model must have 'feature_names_in_' attribute"**
- A: Ensure your model has been fitted before passing it to FastCausalSHAP.

**Q: JSON file validation errors**
- A: Check that your JSON file follows the correct format (see Causal Graph Format section).



## Contributing

Contributions are welcome through pull request!

## Citation
If you use Fast Causal SHAP in your research, please cite:
```
@inproceedings{ng2025causal,
  title={Causal SHAP: Feature Attribution with Dependency Awareness through Causal Discovery},
  author={Ng, Woon Yee and Wang, Li Rong and Liu, Siyuan and Fan, Xiuyi},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2025},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License.
