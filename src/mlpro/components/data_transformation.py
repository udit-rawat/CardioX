import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
import logging
from src.mlpro.entity.config_entity import DataTransformationConfig
logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self):

        data = pd.read_csv(self.config.data_path)
        data.drop(['AgeCategory', 'Race', 'GenHealth'], inplace=True, axis=1)

        X = data.drop('HeartDisease', axis=1)
        y = data['HeartDisease']
        y = y.apply(lambda x: 1 if x == 'Yes' else 0)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply Label Encoding
        X_train, X_test = self.label_encoding(X_train, X_test)

        # Feature Selection using Chi-Square
        X_train_top6, X_test_top6, top6_features = self.feature_selection(
            X_train, X_test, y_train)
        print(f"Top 6 selected features: {list(top6_features)}")

        # Convert selected features back to DataFrame
        X_train_top6 = pd.DataFrame(X_train_top6, columns=top6_features)
        X_test_top6 = pd.DataFrame(X_test_top6, columns=top6_features)

        # Save the transformed train and test data
        pd.concat([X_train_top6, y_train.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(self.config.root_dir, "train.csv"), index=False)
        pd.concat([X_test_top6, y_test.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data preprocessing completed successfully.")
        logger.info(f"Train shape: {X_train_top6.shape}")
        logger.info(f"Test shape: {X_test_top6.shape}")

        return X_train_top6, X_test_top6, y_train, y_test

    def label_encoding(self, X_train, X_test):
        # Apply label encoding to binary columns
        le = LabelEncoder()
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])

        return X_train, X_test

    def feature_selection(self, X_train, X_test, y_train):
        # Feature selection using Chi-Square
        chi_selector = SelectKBest(chi2, k=6)
        X_train_top6 = chi_selector.fit_transform(X_train, y_train)
        X_test_top6 = chi_selector.transform(X_test)

        top6_features = X_train.columns[chi_selector.get_support()]
        return X_train_top6, X_test_top6, top6_features

    def train_test_splitting(self):
        # Preprocess the data and get the train and test sets
        X_train, X_test, y_train, y_test = self.preprocess_data()

        return X_train, X_test, y_train, y_test
