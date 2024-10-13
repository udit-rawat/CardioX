from xgboost import XGBClassifier
import joblib
import pandas as pd
from src.mlpro import logger
import os
from src.mlpro.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop(
            [self.config.target_column], axis=1).to_numpy()
        train_y = train_data[self.config.target_column].values

        xgb = XGBClassifier(
            subsample=self.config.subsample,
            scale_pos_weight=self.config.scale_pos_weight,
            n_estimators=self.config.n_estimators,
            min_child_weight=self.config.min_child_weight,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            colsample_bytree=self.config.colsample_bytree,
            random_state=108)

        xgb.fit(train_x, train_y)

        joblib.dump(xgb, os.path.join(
            self.config.root_dir, self.config.model_name))
