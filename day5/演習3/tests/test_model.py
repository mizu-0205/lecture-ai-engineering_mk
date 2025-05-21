# day5/演習3/train_models.py

import os
import pickle
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ディレクトリ設定
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# データ読み込み or 取得
DATA_PATH = os.path.join(DATA_DIR, "Titanic.csv")
if not os.path.exists(DATA_PATH):
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    df = titanic.data
    df["Survived"] = titanic.target.astype(int)
    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# 特徴量／ターゲット分割
X = df.drop("Survived", axis=1)
y = df["Survived"].astype(int)

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# テスト用と同じ前処理定義
numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 前処理を学習データに適用
X_train_pre = preprocessor.fit_transform(X_train)

# ── ランダムフォレストの学習・保存 ──
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_pre, y_train)

with open(os.path.join(MODEL_DIR, "randomforest_model.pkl"), "wb") as f:
    pickle.dump(rf, f)

# ── ロジスティック回帰の学習・保存 ──
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_pre, y_train)

with open(os.path.join(MODEL_DIR, "logisticregression_model.pkl"), "wb") as f:
    pickle.dump(lr, f)

print("Models have been trained and saved to ../models/")
