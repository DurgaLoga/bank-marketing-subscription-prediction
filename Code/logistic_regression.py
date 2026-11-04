# From Subcription prediction to smart marketing

# Model Baseline - Logitic Regression
import pandas as pd

#to install python -m pip install scikit-learn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

#LOAD DATA
df = pd.read_csv("bank-full.csv", sep=";")
print("Dataset shape:", df.shape)

#CREATE TARGET
y = df["y"].map({
    "no": 0,
    "yes": 1
})

# Predictor variables
X = df.drop(columns=["y"])

print("\nPredictor shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print((y.value_counts(normalize=True) * 100).round(2))

#IDENTIFY FEATURE TYPES
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "str"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of numerical features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))

#PREPROCESSING PIPELINE
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully.")

#we need to used atleast 5-fold cross-validation as per professor's requirement
#so we divide the data into 5 parts
#Because only 11.7% of observations are subscribers, we will use StratifiedKFold 
#This tries to retain the same Yes/No proportion in each fold

#CROSS-VALIDATION AND MCC

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
mcc_scorer = make_scorer(matthews_corrcoef)

print("\n5-fold stratified cross-validation created.")
print("Primary evaluation metric: MCC")

#We haven't split the dataset into train/test sets
#because our requirement specifies 5-fold cross-validation, 
#and we want the same folds and evaluation procedure for our model comparisons

#How well can we do without actually learning anything useful?

#MAJORITY-CLASS BASELINE
dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", dummy_model)
    ]
)

dummy_mcc_scores = cross_val_score(
    dummy_pipeline,
    X,
    y,
    cv=cv,
    scoring=mcc_scorer
)

print("\nMajority-Class Baseline")
print("MCC for each fold:", dummy_mcc_scores)
print("Mean MCC:", dummy_mcc_scores.mean().round(4))

#LOGISTIC REGRESSION - Full Model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", logistic_model)
    ]
)

logistic_mcc_scores = cross_val_score(
    logistic_pipeline,
    X,
    y,
    cv=cv,
    scoring=mcc_scorer
)

print("\nLogistic Regression")
print("MCC for each fold:", logistic_mcc_scores.round(4))
print("Mean MCC:", logistic_mcc_scores.mean().round(4))
print("MCC Standard Deviation:", logistic_mcc_scores.std().round(4))

#LOGISTIC REGRESSION WITHOUT DURATION
X_no_duration = X.drop(columns=["duration"])

numeric_features_no_duration = [
    col for col in numeric_features
    if col != "duration"
]

preprocessor_no_duration = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features_no_duration),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

logistic_no_duration_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor_no_duration),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

logistic_no_duration_mcc = cross_val_score(
    logistic_no_duration_pipeline,
    X_no_duration,
    y,
    cv=cv,
    scoring=mcc_scorer
)

print("\nLogistic Regression WITHOUT Duration")
print(
    "MCC for each fold:",
    logistic_no_duration_mcc.round(4)
)
print(
    "Mean MCC:",
    logistic_no_duration_mcc.mean().round(4)
)
print(
    "MCC Standard Deviation:",
    logistic_no_duration_mcc.std().round(4)
)
