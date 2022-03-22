# Prediction Intervals Explained: A LightGBM Tutorial

When we are performing regression tasks, we have the option to generate prediction intervals, which is not the case when doing classification. We generate prediction intervals using quantile regression, which is a fancy way of estimating the median value for a regression value in a specific quantile.

## Background

LightGBM is one of the fastest and most accurate libraries for regression tasks. To add even more utility into the model, they have implemented prediction intervals for the community to be able to give a range of possible values.

Simply put, a prediction interval is just about generating a lower and upper bound on the final regression value. This is incredibly important for some tasks, which we will further elaborate on now.

### Why use them?

We can never be 100% certain about one prediction from one model, so instead, the idea is to give an interval back to a person who ends up controlling the final decision based on the range given by the model. For example, if we are trying to set the price for a house, it is common knowledge that the price is incredibly dependent on how well maintained and renovated it is.

Therefore, we want to give a range - and if the house is poorly maintained, perhaps the price would land in the lower end of the price interval.

### Quantile Regression Explained

In the typical linear regression model, we are tracking the mean difference from the ground truth to optimize the model. However, in quantile regression, as the name suggests, we track a specific quantile (also know as a percentile) against the median of the ground truth.

This specific approach enables us to specify the quantiles. For example, we most often specify that we want the 5% quantile (covering 5% of the data) and the 95% quantile (covering 95% of the data). This gives us a lower and upper boundary that we can use as our smallest and highest estimate in a regression task.

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
