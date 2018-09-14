import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

wine_data = pd.read_csv('wine_quality_train.csv')
test_data = pd.read_csv('wine_quality_test.csv')

correlation = wine_data.corr()
fig = plt.subplots(figsize=(10, 10), dpi=300)
#sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')
#plt.show()

features = ['fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates',
            'alcohol'] # 11 features
x = np.array(wine_data[features])
y = np.array(wine_data['quality'])
test_x = np.array(test_data[features])
test_y = np.array(test_data['quality'])


regress_model = LinearRegression()
regress_model.fit(x, y)
y_pred = regress_model.predict(test_x)
rmse = mean_squared_error(test_y, y_pred)

r2 = r2_score(test_y, y_pred)
print(np.round(regress_model.coef_, decimals=4))
print(rmse)  # 0.6266776206788026
print(r2)  # 0.4057082915125355

y_range = np.arange(len(regress_model.coef_))
plt.bar(y_range, regress_model.coef_, align='center', alpha=0.5)
for a,b in zip(y_range, regress_model.coef_):
  plt.text(a, b, np.round(b, decimals=4), horizontalalignment='center')
plt.xticks(y_range, features, rotation=30)
plt.ylabel('Linear Regression Model Parameters')
plt.show()

