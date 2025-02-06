import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap
from sklearn.feature_selection import RFE
import optuna
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN

# Load dataset (Titanic dataset as an example)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Feature Engineering: Interaction & Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(df[['Fare', 'Pclass', 'Age', 'SibSp']])
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out(['Fare', 'Pclass', 'Age', 'SibSp']))
df = pd.concat([df, interaction_df], axis=1)

# Outlier Detection with Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(df.drop(columns=['Survived']))
df = df[outliers == 1]

# Outlier Detection with DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_outliers = dbscan.fit_predict(df.drop(columns=['Survived']))
df = df[dbscan_outliers != -1]

# Feature Selection with SHAP
X = df.drop(columns=['Survived'])
y = df['Survived']
model = XGBClassifier()
model.fit(X, y)
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap_importances = np.abs(shap_values.values).mean(axis=0)
important_features = X.columns[np.argsort(shap_importances)[-10:]]  # Select top 10 features
X = X[important_features]

# Recursive Feature Elimination (RFE)
rfe = RFE(estimator=XGBClassifier(), n_features_to_select=5)
rfe.fit(X, y)
X = X.loc[:, rfe.support_]

# Hyperparameter Tuning with Bayesian Optimization
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params

# Train Final Model
final_model = XGBClassifier(**best_params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, preds)
print(f"Final Model Accuracy: {accuracy:.4f}")
