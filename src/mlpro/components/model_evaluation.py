import os
import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, matthews_corrcoef
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
import json
from src.mlpro.entity.config_entity import ModelEvaluationConfig
from src.mlpro.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_classification_metrics(self, actual, pred, pred_proba):
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred_proba)
        mcc = matthews_corrcoef(actual, pred)
        return precision, recall, f1, roc_auc, mcc

    def log_into_mlflow(self):
        # Load the model and data
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        test_x = test_data.drop([self.config.target_column], axis=1).to_numpy()
        test_y = test_data[self.config.target_column].values

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            # Make predictions using XGBoost
            y_pred_proba = model.predict_proba(test_x)[:, 1]  # Probabilities
            y_pred = (y_pred_proba >= 0.5).astype(
                int)        # Binary predictions

            # Evaluate classification metrics
            precision, recall, f1, roc_auc, mcc = self.eval_classification_metrics(
                test_y, y_pred, y_pred_proba)

            # Save metrics as local
            scores = {
                "precision": precision, "recall": recall, "f1_score": f1,
                "roc_auc": roc_auc, "mcc": mcc
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log all parameters and classification metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("mcc", mcc)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="XGBClassifier"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
