from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# iris = load_iris()
# X, y = iris.data, iris.target

# logreg_model = LogisticRegression(max_iter=200)
# logreg_model.fit(X, y)

# rf_model = RandomForestClassifier()
# rf_model.fit(X, y)

# joblib.dump(logreg_model, 'logistic_regression_model.pkl')

# joblib.dump(rf_model, 'random_forest_model.pkl')

# print("Models saved successfully.")

# import joblib
# import os
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train models
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X, y)

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Save models in the main directory
logreg_model_path = "logistic_regression_model.pkl"
rf_model_path = "random_forest_model.pkl"

joblib.dump(logreg_model, logreg_model_path)
joblib.dump(rf_model, rf_model_path)

print("Models saved successfully in the main directory.")
