import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import random
import pickle
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature


# データ準備
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = "data/Titanic.csv"
    data = pd.read_csv(path)

    # 必要な特徴量の選択と前処理
    data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])  # 性別を数値に変換

    # 整数型の列を浮動小数点型に変換
    data["Pclass"] = data["Pclass"].astype(float)
    data["Sex"] = data["Sex"].astype(float)
    data["Age"] = data["Age"].astype(float)
    data["Fare"] = data["Fare"].astype(float)
    data["Survived"] = data["Survived"].astype(float)

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# 学習と評価
def train_and_evaluate(
    X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42
):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=random.randint(50, 200),
            max_depth=random.choice([None, 3, 5, 10, 15]),
            random_state=random_state,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=random_state
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name} の精度: {acc:.4f}")
        results[name] = {"model": model, "accuracy": acc, "params": model.get_params()}
    return results


# モデル保存
def log_model(model, accuracy, params, X_train, X_test, model_name):
    with mlflow.start_run():
        # パラメータをログ
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        # モデルのハイパーパラメータをログ
        mlflow.log_param("model_name", model_name)
        # メトリクスをログ
        mlflow.log_metric("accuracy", accuracy)

        # モデルのシグネチャを推論
        signature = infer_signature(X_train, model.predict(X_train))

        # モデルを保存
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:5],  # 入力例を指定
        )
        # accurecyとparmsは改行して表示
        print(f"モデルのログ記録値 \naccuracy: {accuracy}\nparams: {params}")


# メイン処理
if __name__ == "__main__":
    # ランダム要素の設定
    test_size = round(
        random.uniform(0.1, 0.3), 2
    )  # 10%〜30%の範囲でテストサイズをランダム化
    data_random_state = random.randint(1, 100)
    model_random_state = random.randint(1, 100)
    n_estimators = random.randint(50, 200)
    max_depth = random.choice([None, 3, 5, 10, 15])

    # パラメータ辞書の作成
    params = {
        "test_size": test_size,
        "data_random_state": data_random_state,
        "model_random_state": model_random_state,
        "n_estimators": n_estimators,
        "max_depth": "None" if max_depth is None else max_depth,
    }

    # データ準備
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=test_size, random_state=data_random_state
    )
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test, random_state=model_random_state
    )
    # 学習と評価
    os.makedirs("models", exist_ok=True)
    for name, res in results.items():
        log_model(
            res["model"],
            res["accuracy"],
            res["params"],
            X_train,
            X_test,
            model_name=name,
        )
        model_path = os.path.join("models", f"{name.lower()}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(res["model"], f)
        print(f"{name} モデルを {model_path} に保存しました")
