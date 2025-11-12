

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def load_data():
    if os.path.exists("train.csv"):
        print("Loading train.csv and (if exists) test.csv from current directory...")
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv") if os.path.exists("test.csv") else None
        return train, test
    try:
        tit = sns.load_dataset("titanic")
        tit = tit.dropna(subset=["survived"]).copy()
        tit.rename(columns={'survived': 'Survived', 'pclass': 'Pclass', 'sex': 'Sex', 'age': 'Age', 'fare': 'Fare', 'embarked': 'Embarked', 'sibsp': 'SibSp', 'parch': 'Parch', 'who': 'Who', 'deck': 'Deck', 'alone': 'Alone'}, inplace=True)
        tit['PassengerId'] = np.arange(1, len(tit) + 1)
        print("Loaded seaborn titanic dataset (no separate test set).\n")
        return tit, None
    except Exception:
        raise FileNotFoundError("No train.csv found and seaborn dataset unavailable.")


def feature_engineering(df):
    df = df.copy()
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
        rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    else:
        df['Title'] = 'Unknown'

    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    else:
        df['FamilySize'] = 1
        df['IsAlone'] = 1

    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notnull().astype(int)
        df['Deck'] = df['Cabin'].astype(str).str[0].replace('n', np.nan)
    else:
        df['HasCabin'] = 0
        df['Deck'] = np.nan

    return df


def preprocess_and_get_pipeline(df):
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64] and c not in ('PassengerId', 'Survived')]
    keep_numeric = [c for c in ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize'] if c in numeric_cols]
    categorical_cols = [c for c in df.columns if df[c].dtype == object and c not in ('Name', 'Ticket', 'Cabin')]
    for c in ['Sex', 'Embarked', 'Title', 'Deck']:
        if c in df.columns and c not in categorical_cols:
            categorical_cols.append(c)

    print("Numeric features:", keep_numeric)
    print("Categorical features:", categorical_cols)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, keep_numeric),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor, keep_numeric, categorical_cols


def build_and_train(train_df):
    df = feature_engineering(train_df)

    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    use_cols = [c for c in df.columns if c not in drop_cols]

    if 'Survived' not in df.columns:
        raise ValueError("Training dataframe must contain 'Survived' column")

    X = df[use_cols].drop(columns=['Survived'])
    y = df['Survived'].astype(int)

    preprocessor, keep_numeric, categorical_cols = preprocess_and_get_pipeline(X)

    pipe_lr = Pipeline(steps=[('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
    pipe_rf = Pipeline(steps=[('pre', preprocessor), ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))])

    grid_lr = {'clf__C': [0.01, 0.1, 1, 10]}
    grid_rf = {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10, None]}

    print("Starting GridSearch for Logistic Regression...")
    gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
    gs_lr.fit(X, y)
    print("Best LR params:", gs_lr.best_params_)
    print("LR CV score:", gs_lr.best_score_)

    print("Starting GridSearch for Random Forest...")
    gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    gs_rf.fit(X, y)
    print("Best RF params:", gs_rf.best_params_)
    print("RF CV score:", gs_rf.best_score_)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_model = gs_rf if gs_rf.best_score_ >= gs_lr.best_score_ else gs_lr
    print("Selected best model based on CV: ", 'RandomForest' if best_model is gs_rf else 'LogisticRegression')

    best_model.refit = True
    best_model.best_estimator_.fit(X, y)

    y_pred = best_model.best_estimator_.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, y_pred))
    print("Classification report:\n", classification_report(y_val, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

    # === Survival Visualization Section ===
    print("\nGenerating survival graphs...\n")
    try:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Survived', data=df)
        plt.title('Overall Survival Count')
        plt.show()

        if 'Sex' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x='Sex', hue='Survived', data=df)
            plt.title('Survival by Gender')
            plt.show()

        if 'Pclass' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x='Pclass', hue='Survived', data=df)
            plt.title('Survival by Passenger Class')
            plt.show()

        if 'Age' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df, x='Age', hue='Survived', multiple='stack', bins=20)
            plt.title('Survival by Age Distribution')
            plt.show()
    except Exception as e:
        print(f"Graph generation failed: {e}")

    joblib.dump(best_model.best_estimator_, 'titanic_best_model.joblib')
    print("Saved best model to 'titanic_best_model.joblib'")

    return best_model.best_estimator_, preprocessor, keep_numeric, categorical_cols


def make_submission(model, test_df, preprocessor, keep_numeric, categorical_cols):
    df_test = feature_engineering(test_df)
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    use_cols = [c for c in df_test.columns if c not in drop_cols]

    X_test = df_test[use_cols]
    preds = model.predict(X_test)

    submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': preds.astype(int)})
    submission.to_csv('submission.csv', index=False)
    print("Saved submission to submission.csv")


if __name__ == '__main__':
    train_df, test_df = load_data()
    if test_df is None:
        print("No test.csv provided; splitting the available data into train/val for demonstration.")
        model, preproc, knum, kcat = build_and_train(train_df)
    else:
        model, preproc, knum, kcat = build_and_train(train_df)
        if test_df is not None:
            print("Making submission using provided test.csv...")
            make_submission(model, test_df, preproc, knum, kcat)

    print("Done.")
