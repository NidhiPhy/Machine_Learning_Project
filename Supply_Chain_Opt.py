import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('/Users/alfredoserranofigueroa/Desktop/Github/Supply_Chain_Optimization/features.csv')
stores = pd.read_csv('/Users/alfredoserranofigueroa/Desktop/Github/Supply_Chain_Optimization/stores.csv')
test = pd.read_csv('/Users/alfredoserranofigueroa/Desktop/Github/Supply_Chain_Optimization/test.csv')
train = pd.read_csv('/Users/alfredoserranofigueroa/Desktop/Github/Supply_Chain_Optimization/train.csv')

data = train.merge(stores, how='left', on='Store')
data = data.merge(features, how='left', on=['Store', 'Date'])

data['Date'] = pd.to_datetime(data['Date'])

data = data.sort_values(by='Date')

numeric_columns = data.select_dtypes(include=[np.number]).columns

data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

data = data.fillna(data.mode().iloc[0])

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek

categorical_columns = data.select_dtypes(include=['object']).columns

if len(categorical_columns) > 0:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

X = data.drop(columns=['Weekly_Sales', 'Date'])
y = data['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 8))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

fig = px.line(data, x='Date', y='Weekly_Sales', title='Sales Trend Over Time', labels={'Date': 'Date', 'Weekly_Sales': 'Weekly Sales'})
fig.update_layout(xaxis_title='Date', yaxis_title='Weekly Sales')
fig.show()

store_type_columns = [col for col in data.columns if 'StoreType' in col]
if len(store_type_columns) > 0:
    fig = px.box(data, x=store_type_columns[0], y='Weekly_Sales', title='Sales Distribution by Store Type', labels={'Weekly_Sales': 'Weekly Sales'})
    fig.update_layout(xaxis_title='Store Type', yaxis_title='Weekly Sales')
    fig.show()

fig = px.box(data, x='Month', y='Weekly_Sales', title='Sales Distribution by Month', labels={'Month': 'Month', 'Weekly_Sales': 'Weekly Sales'})
fig.update_layout(xaxis_title='Month', yaxis_title='Weekly Sales')
fig.show()

fig = px.box(data, x='DayOfWeek', y='Weekly_Sales', title='Sales Distribution by Day of the Week', labels={'DayOfWeek': 'Day of Week', 'Weekly_Sales': 'Weekly Sales'})
fig.update_layout(xaxis_title='Day of the Week', yaxis_title='Weekly Sales')
fig.show()

