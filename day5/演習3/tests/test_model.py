import os
import time
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_FILES = [
    os.path.join(MODEL_DIR, "randomforest_model.pkl"),
    os.path.join(MODEL_DIR, "logisticregression_model.pkl"),
]


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Survived",
            ]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def _load_model(path: str):
    """pickle からモデルを読み込む"""
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("model_path", MODEL_FILES)
def test_model_exists(model_path):
    """モデルファイルが存在するか確認"""
    if not os.path.exists(model_path):
        pytest.skip(f"{model_path} が存在しないためスキップします")
    assert os.path.exists(model_path), f"{model_path} が存在しません"


@pytest.mark.parametrize("model_path", MODEL_FILES)
def test_model_accuracy_and_time(sample_data, preprocessor, model_path):
    """モデルの精度と推論時間を検証"""
    model = _load_model(model_path)

    # テスト用データの準備
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 精度チェック
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.75, f"{os.path.basename(model_path)} の精度が低すぎます: {acc:.4f}"

    # 推論時間チェック
    start = time.time()
    model.predict(X_test)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"{os.path.basename(model_path)} の推論時間が長すぎます: {elapsed:.3f}s"


def test_model_reproducibility(sample_data, preprocessor):
    """RandomForest モデルの再現性を検証"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def make_rf():
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                    ),
                ),
            ]
        )

    model1 = make_rf()
    model2 = make_rf()
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    assert np.array_equal(
        pred1, pred2
    ), "RandomForest モデルの予測結果に再現性がありません"


