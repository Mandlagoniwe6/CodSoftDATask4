import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading data
salesData = pd.read_csv("advertising (1).csv",header = 0, sep = ",")

# The relationships between advertising channels and sales
sns.pairplot(salesData, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=0.8, kind='scatter')
plt.suptitle('Relationships Between Advertising Channels and Sales', fontsize=14)
plt.show()

#The relationship between TV advertising and Sales
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=salesData['TV'], y=salesData['Sales'], color='blue', alpha=0.7)
plt.title('Relationship Between TV Advertising and Sales', fontsize=14)
plt.xlabel('TV Advertising Expenditure', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.show()

# Preparing data for regression
X = salesData[['TV']]  # Independent variable (TV advertising expenditure)
y = salesData['Sales']  # Dependent variable (Sales)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# The regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(x=salesData['TV'], y=salesData['Sales'], color='blue', alpha=0.7, label='Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: TV Advertising vs Sales', fontsize=14)
plt.xlabel('TV Advertising Expenditure', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.legend()
plt.show()

# Actual vs Predicted sales
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Sales', fontsize=14)
plt.xlabel('Actual Sales', fontsize=12)
plt.ylabel('Predicted Sales', fontsize=12)
plt.show()

# Residuals vs. Predicted values
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, color='purple', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)  
plt.title('Residuals vs Predicted Sales', fontsize=14)
plt.xlabel('Predicted Sales', fontsize=12)
plt.ylabel('Residuals (Errors)', fontsize=12)
plt.show()

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(f"Model Accuracy Score (R² as %): {r2 * 100:.2f}%")
