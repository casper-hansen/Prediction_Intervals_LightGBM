# Prediction Intervals Explained: A LightGBM Tutorial

When we are performing regression tasks, we have the option to generate prediction intervals, which is not the case when doing classification. We generate prediction intervals using quantile regression.

## Background

### Why use them?

## Python Example

To make prediction intervals, we need a lower bound and upper bound for the prediction we generate using our model. To generate these bounds, we use the following method:

1. Choose a prediction interval, usually we set it to 95% or 0.95 — we call this the alpha parameter ($\alpha$) when making prediction intervals.
2. Train your model for making predictions on your dataset.
3. Train two models, one for the lower bound and another for the upper bound. We need to set two parameters for this to work: objective and alpha.

### Data Loading & Preparation

Packages

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
```

Data loader function

```python

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y
```

Call data loader and split data

```python
x, y = sklearn_to_df(fetch_california_housing())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
```

### Prediction Intervals

```python
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data_loader import x_train, x_test, y_train, y_test
```

```python
regressor = lgb.LGBMRegressor()
regressor.fit(x_train, y_train)
regressor_pred = regressor.predict(x_test)
```

```python
lower = lgb.LGBMRegressor(objective = 'quantile', alpha = 1 - 0.95)
lower.fit(x_train, y_train)
lower_pred = lower.predict(x_test)
```

```python
upper = lgb.LGBMRegressor(objective = 'quantile', alpha = 0.95)
upper.fit(x_train, y_train)
upper_pred = upper.predict(x_test)
```

```python
score = r2_score(y_test, regressor_pred)
print(score)
```

```python
plt.figure(figsize=(10, 6))

plt.scatter(x_test.MedInc, lower_pred, color='limegreen', marker='o', label='lower', lw=0.5, alpha=0.5)
plt.scatter(x_test.MedInc, regressor_pred, color='aqua', marker='x', label='pred', alpha=0.7)
plt.scatter(x_test.MedInc, upper_pred, color='dodgerblue', marker='o', label='upper', lw=0.5, alpha=0.5)
plt.plot(sorted(x_test.MedInc), sorted(lower_pred), color='black')
plt.plot(sorted(x_test.MedInc), sorted(regressor_pred), color='red')
plt.plot(sorted(x_test.MedInc), sorted(upper_pred), color='black')
plt.legend()

plt.show()
```
