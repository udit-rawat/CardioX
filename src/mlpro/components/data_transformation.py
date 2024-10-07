import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from src.mlpro.entity.config_entity import DataTransformationConfig
logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def preprocess_data(self):
        # Load the data
        data = pd.read_csv(self.config.data_path)

        # Drop columns that won't be used (already encoded)
        data.drop(['AgeCategory', 'Race', 'GenHealth'], inplace=True, axis=1)

        # Split the data into training and test sets
        train, test = train_test_split(
            data, test_size=0.20, random_state=42, stratify=data['HeartDisease'])  # Stratify by target

        # Preprocess train and test separately to avoid data leakage
        X_train, y_train = train.drop(
            'HeartDisease', axis=1), train['HeartDisease']
        X_test, y_test = test.drop(
            'HeartDisease', axis=1), test['HeartDisease']

        # Transform train and test sets
        X_train, X_test = self.transform_data(X_train, X_test)

        # Save the transformed train and test data
        pd.concat([X_train, y_train], axis=1).to_csv(
            os.path.join(self.config.root_dir, "train.csv"), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(
            os.path.join(self.config.root_dir, "test.csv"), index=False)

        return X_train, X_test, y_train, y_test

    def transform_data(self, X_train, X_test):
        # Apply label encoding to binary columns
        X_train, X_test = self.label_encoding(X_train, X_test)

        # Apply standard scaling to continuous columns
        X_train, X_test = self.standard_scaling(X_train, X_test)

        return X_train, X_test

    def label_encoding(self, X_train, X_test):
        # Binary columns to encode
        binary_cols = [
            'Smoking',
            'AlcoholDrinking',
            'Stroke',
            'DiffWalking',
            'Sex',
            'Diabetic',
            'PhysicalActivity',
            'Asthma',
            'KidneyDisease',
            'SkinCancer'
        ]

        label_encoder = LabelEncoder()
        for col in binary_cols:
            X_train[col] = label_encoder.fit_transform(X_train[col])
            # Apply the same transformation on test set
            X_test[col] = label_encoder.transform(X_test[col])

        return X_train, X_test

    def standard_scaling(self, X_train, X_test):
        # Ensure train and test sets have the same columns
        X_train, X_test = X_train.align(
            X_test, join='left', axis=1, fill_value=0)

        # Standardize continuous columns
        continuous_cols = ['BMI', 'PhysicalHealth',
                           'MentalHealth', 'SleepTime']
        scaler = StandardScaler()

        X_train[continuous_cols] = scaler.fit_transform(
            X_train[continuous_cols])
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

        return X_train, X_test

    def train_test_splitting(self):
        # Preprocess the data and get the train and test sets
        X_train, X_test, y_train, y_test = self.preprocess_data()

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test shape: {X_test.shape}")

        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test
