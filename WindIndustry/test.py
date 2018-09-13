import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

wine_data = pd.read_csv('wine_quality_train.csv')

correlation = wine_data.corr()
fig = plt.subplots(figsize=(10, 10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')
#plt.show()

features = ['fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates',
            'alcohol'] # 11 features
x = np.array(wine_data[features])
y = np.array(wine_data['quality'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)

regress_model = LinearRegression()
regress_model.fit(x_train, y_train)
y_pred = regress_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = regress_model.score(x_test, y_test)
print(np.round(regress_model.coef_, decimals=4))
print(rmse)  # 0.6266776206788026
print(r2)  # 0.4057082915125355

