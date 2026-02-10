import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# --- LOAD DATA ---
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)

# print(df.head())

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('charges', axis=1) # Use ALL features (age, bmi, children, smoker_yes, etc.)
y = df_encoded['charges']

# features = ["age", "bmi"]

# X = df[features]
# y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- TRAIN  MODEL ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- EVALUATION ---
y_pred = model.predict(X_test)

print("--- Model Performance ---")
print(f"R2: {metrics.r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f}")
print("-" * 40)

# --- COEFFICIENTS ---
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Impact ($)'])
print(coeff_df)

# --- VISUALIZE Graph ---
plt.figure(figsize=(8, 5))
sns.barplot(x=coeff_df.index, y=coeff_df['Impact ($)'], hue=coeff_df.index, palette='viridis', legend=False)
plt.title('What drives patient spending?')
plt.ylabel('Increase in Spending ($)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()

# --- PREDICTION ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual patient spending ($)')
plt.ylabel('Predicted patient spending ($)')
plt.title('Actual vs. Predicted')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- INTERCEPT (math anchor) ---
print(f"Base Price (Intercept): ${model.intercept_:.2f}")