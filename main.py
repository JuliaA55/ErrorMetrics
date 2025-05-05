import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X) + 0.1 * X**2

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

X_plot = np.linspace(-20, 20, 400).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_true = np.sin(X_plot) + 0.1 * X_plot**2
y_model = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.plot(X_plot, y_true, label="Реальна функція", color='blue')
plt.plot(X_plot, y_model, label="Прогноз моделі", color='red', linestyle='--')
plt.title("Поліноміальна регресія: f(x) = sin(x) + 0.1·x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
