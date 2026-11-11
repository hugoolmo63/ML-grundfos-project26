from pathlib import Path 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Main paths:
data_p = Path('data')

case1_p = data_p/'case1'
test1_p = case1_p/'test1.csv'
train1_p = case1_p/'train1.csv'

# Main data frames:
test1 = pd.read_csv(test1_p)
train1 = pd.read_csv(train1_p)

# Adding KRPM column:
test1['speed_krpm'] = test1['speed_rpm'] / 1000
train1['speed_krpm'] = train1['speed_rpm'] / 1000 

# Polynomial features for Task 0:
train_features = pd.DataFrame(index=train1.index)
for i in range(4):
    for j in range(4):
        if i == 0 and j == 0:
            continue  # The intercept is handled by LinearRegression
        train_features[f'n{i}_Q{j}'] = train1['speed_krpm']**i * train1['flow_m3_h']**j


# Select polynomial features with forward selection and 5-fold CV:
X = train_features
y = train1['power_w']
cv = KFold(n_splits=5, shuffle=True, random_state=42)

selected_features = []
remaining_features = list(X.columns)
best_rmse = float('inf')

while remaining_features:
    candidate_scores = []
    for feature in remaining_features:
        candidate = selected_features + [feature]
        scores = cross_val_score(
            LinearRegression(), X[candidate], y,
            cv=cv, scoring='neg_root_mean_squared_error'
        )
        candidate_scores.append((-scores.mean(), feature))

    candidate_rmse, candidate_feature = min(candidate_scores)
    if candidate_rmse >= best_rmse:
        break  # Stop when adding another feature does not improve CV RMSE

    selected_features.append(candidate_feature)
    remaining_features.remove(candidate_feature)
    best_rmse = candidate_rmse
    print(f'{len(selected_features):2d} features | CV RMSE: {best_rmse:.4f} | Added: {candidate_feature}')

# Fit the selected model on the full training set:
model = LinearRegression()
model.fit(X[selected_features], y)

print('Selected features:', selected_features)
print(f'Best CV RMSE: {best_rmse:.4f}')

for feature, coefficient in zip(selected_features, model.coef_):
    print(f'{feature}: {coefficient:.6f}')

# Print the fitted polynomial equation:
equation = f'P = {model.intercept_:.6f}'
for feature, coefficient in zip(selected_features, model.coef_):
    i, j = map(int, feature[1:].split('_Q'))
    term = ''.join(part for part, exponent in [('n', i), ('Q', j)]
                   if exponent > 0
                   for part in [part if exponent == 1 else f'{part}^{exponent}'])
    sign = '+' if coefficient >= 0 else '-'
    equation += f' {sign} {abs(coefficient):.6f}*{term}'
print('\nFitted power equation (n in kRPM, Q in m³/h):')
print(equation)

# Evaluate selected polynomial model on test data:
test_features = pd.DataFrame(index=test1.index)
for i in range(4):
    for j in range(4):
        if i == 0 and j == 0:
            continue
        test_features[f'n{i}_Q{j}'] = test1['speed_krpm']**i * test1['flow_m3_h']**j

X_test = test_features[selected_features]
y_test = test1['power_w']
y_pred = model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print(f'Test RMSE: {test_rmse:.4f} W')
print(f'Test R²: {test_r2:.4f}')

# Recover the fitted polynomial coefficients:
print('\nRecovered power equation coefficients:')
print(f'Intercept (n0_Q0): {model.intercept_:.6f}')
for feature, coefficient in zip(selected_features, model.coef_):
    print(f'{feature}: {coefficient:.6f}')

# Plot prediction errors against flow:
import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(test1['flow_m3_h'], residuals, alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Flow (m³/h)')
plt.ylabel('Prediction error (W)')
plt.title('Task 0: Power prediction errors vs flow')
plt.tight_layout()
plt.show()