# Prediction Intervals Explained: A LightGBM Tutorial

When we are performing regression tasks, we have the option to generate prediction intervals, which is not the case when doing classification. We generate prediction intervals using quantile regression.

## Background

### Why use them?

## Python Example

To make prediction intervals, we need a lower bound and upper bound for the prediction we generate using our model. To generate these bounds, we use the following method:

1. Choose a prediction interval, usually we set it to 95% or 0.95 — we call this the alpha parameter ($\alpha$) when making prediction intervals.
2. Train your model for making predictions on your dataset.
3. Train two models, one for the lower bound and another for the upper bound. We need to set two parameters for this to work: objective and alpha.

We can make an upper bound model by 

```python
lgb.LGBMRegressor(objective = 'quantile', alpha = 0.95)
```

We can also make the lower bound model by

```python
lgb.LGBMRegressor(objective = 'quantile', alpha = 1 - 0.95)
```
