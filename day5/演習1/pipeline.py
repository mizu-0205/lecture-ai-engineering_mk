from kedro.io import MemoryDataset, KedroDataCatalog
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import random
import logging
from typing import Dict  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# データ準備 ─ 変更なし
def prepare_data():
    path = "data/Titanic.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = pd.read_csv(path)[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    for col in ["Pclass", "Sex", "Age", "Fare", "Survived"]:
        data[col] = data[col].astype(float)
    X, y = data[["Pclass", "Sex", "Age", "Fare"]], data["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 学習と評価 ─ 複数モデル対応
def train_and_evaluate(X_train, X_test, y_train, y_test):
    seed = random.randint(1, 100) 
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=random.randint(50, 200),
            max_depth=random.choice([None, 3, 5, 10, 15]),
            random_state=seed,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=seed
        ),
    }

    accuracies, params_dict = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies[name] = acc
        params_dict[name] = model.get_params()
        logger.info(f"[{name}] accuracy: {acc:.4f}")
    return models, accuracies, params_dict, seed  


# モデル保存 ─ 複数モデル分ループ
def log_models(models: Dict, accuracies: Dict, params_dict: Dict, seed, X_train, X_test):
    mlflow.set_experiment("titanic-survival-prediction")
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("seed", seed)
            mlflow.log_params(params_dict[name])
            mlflow.log_metric("accuracy", accuracies[name])

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=X_test.iloc[:5]
            )
            logger.info(f"[{name}] モデルを MLflow に記録しました")


# Kedro パイプライン定義
def create_pipeline():
    return Pipeline(
        [
            node(
                prepare_data,
                inputs=None,
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="prepare_data",
            ),
            node(
                train_and_evaluate,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs=["models", "accuracies", "params_dict", "seed"],  
                name="train_and_evaluate",
            ),
            node(
                log_models,  # ★ 関数名変更
                inputs=["models", "accuracies", "params_dict", "seed", "X_train", "X_test"],
                outputs=None,
                name="log_models",
            ),
        ]
    )


if __name__ == "__main__":
    pipeline = create_pipeline()
    catalog = KedroDataCatalog(
        {
            "X_train": MemoryDataset(),
            "X_test": MemoryDataset(),
            "y_train": MemoryDataset(),
            "y_test": MemoryDataset(),
            "models": MemoryDataset(),       
            "accuracies": MemoryDataset(),  
            "params_dict": MemoryDataset(),  
            "seed": MemoryDataset(),    
        }
    )
    logger.info("パイプラインの実行を開始します。")
    SequentialRunner().run(pipeline, catalog)
    logger.info("パイプラインの実行が完了しました。")

