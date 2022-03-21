import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data_loader import x_train, x_test, y_train, y_test

# mean prediction
regressor = lgb.LGBMRegressor()
regressor.fit(x_train, y_train)
regressor_pred = regressor.predict(x_test)

# train lower bound model
lower = lgb.LGBMRegressor(objective = 'quantile', alpha = 1 - 0.95)
lower.fit(x_train, y_train)
lower_pred = lower.predict(x_test)

# train upper bound model
upper = lgb.LGBMRegressor(objective = 'quantile', alpha = 0.95)
upper.fit(x_train, y_train)
upper_pred = upper.predict(x_test)

score = r2_score(y_test, regressor_pred)
print(score)

plt.figure(figsize=(10, 6))

plt.scatter(x_test.MedInc, lower_pred, color='limegreen', marker='o', label='lower', lw=0.5, alpha=0.5)
plt.scatter(x_test.MedInc, regressor_pred, color='aqua', marker='x', label='pred', alpha=0.7)
plt.scatter(x_test.MedInc, upper_pred, color='dodgerblue', marker='o', label='upper', lw=0.5, alpha=0.5)
plt.plot(sorted(x_test.MedInc), sorted(lower_pred), color='black')
plt.plot(sorted(x_test.MedInc), sorted(regressor_pred), color='red')
plt.plot(sorted(x_test.MedInc), sorted(upper_pred), color='black')
plt.legend()

plt.show()