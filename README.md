# ml-fusion-evaluation

This Project is to build a Generic Evaluation Framework which takes in inputs such as Model, 
Features and outputs/tracks items like Metrics, Hyperparameters. 


# How to use

## Regression Model Evaluation

```python
from fusion_evaluation.metrics import regression

# Basket Prediction
y_pred = [9,10,15,8]
y = [10,12,15,11]
re = regression.RegressionEval(y_pred=y_pred, y_true=y, num_features=10)
eval_results_bp = regression.metrics(re)

print(eval_results_bp)

#output
RegressionMetrics(mean_squared_error=3.5, mean_absolute_error=1.5, r_squared=0.0, r_square_adjusted=-0.5)
```

## Classification Model Evaluation

```python
from fusion_evaluation.metrics import classification

y_pred = [1,0,1,0,1,0,1,0,1,1]
y = [1,1,1,0,1,1,1,0,1,1]
ce = classification.ClassificationEval(y_pred=y_pred, y_true=y)
eval_results_ce = classification.metrics(ce)

print(eval_results_ce.__dict__)
{'accuracy': 0.8,
 'auc_roc': 0.875,
 'auc_prc': 0.95,
 'precision': 1.0,
 'recall': 0.75,
 'f1score': 0.8571428571428571,
 'roc': Roc(fpr=[0.0, 0.0, 1.0], tpr=[0.0, 0.75, 1.0], thr_roc=[2, 1, 0]),
 'prc': Prc(precision_thr=[0.8, 1.0, 1.0], recall_thr=<built-in method tolist of numpy.ndarray object at 0x136c69d00>, thr_prc=[0, 1]),
 'fvr': Fvr(fraction=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], recall_fvr=[0.125, 0.125, 0.25, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], thr_fvr=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1])}
```
