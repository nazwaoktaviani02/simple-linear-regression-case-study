# Import Necessary Libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Input Data
hours_studied = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Independent Variable (X)
final_scores = np.array([50, 55, 60, 65, 70]) # Dependent Variable (Y)

# Create and Fit The Model
model = LinearRegression()
model.fit(hours_studied, final_scores)

# Predicting for 6 Hours of Study
predicted_score = model.predict([[6]])


# Model Parameters
slope = model.coef_[0]
intercept = model.intercept_

# Plotting The Results
plt.scatter(hours_studied, final_scores, color='green')
plt.plot(hours_studied, model.predict(hours_studied), color='red')
plt.title('Simple Linear Regresion: Hours Studied vs Final Score')
plt.xlabel('Hours Studied')
plt.ylabel('Final Score')
plt.show()

print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")
print(f"Predicted score for 6 hours of study: {predicted_score[0]}")
