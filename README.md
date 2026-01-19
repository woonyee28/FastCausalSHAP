# Fast Causal SHAP

Fast Causal SHAP is a Python package designed for efficient and interpretable SHAP value computation in causal inference tasks. It integrates seamlessly with various causal inference frameworks and enables feature attribution with awareness of causal dependencies.

## Features

- Fast computation of SHAP values for causal models
- Support for multiple causal inference frameworks

## Installation

Install the stable version via PyPI:

```bash
pip install fast-causal-shap
```

Or, for the latest development version:

```bash
pip install git+https://github.com/woonyee28/CausalSHAP.git
```

## Usage
```
from fast_causal_shap.core import FastCausalSHAP

# Predict probabilities and assign to training data
predicted_probabilities = model.predict_proba(X_train)[:,1]
X_train['target'] = predicted_probabilities

# Initialize FastCausalInference
ci = FastCausalInference(data=X_train, model=model, target_variable='target')

# Load causal strengths (precomputed using R packages)
ci.load_causal_strengths(result_dir + 'Causal_Effect.json')

# Compute modified SHAP values for a single instance
x_instance = X_train.iloc[33]

print(ci.compute_modified_shap_proba(x_instance, is_classifier=True))
```

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

Fast Causal SHAP supports integration with structural algorithms such as:
1. Peter-Clarke (PC) Algorithm
2. IDA Algorithm
3. Fast Causal Inference (FCI) Algorithm
You can find example R code for these integrations here: [FastCausalSHAP R code examples](https://github.com/woonyee28/CausalSHAP/tree/main/code/r)


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
