import joblib
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt


class PredictionPipeline:
    def __init__(self):
        # Load the pre-trained model during initialization
        self.model = joblib.load(
            Path('artifacts/model_trainer/model.joblib'))
        # Initialize the SHAP explainer based on the loaded model
        self.explainer = shap.Explainer(self.model)

    def predict_and_explain(self, data):
        # Ensure data is reshaped to handle a single instance (if needed)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Make predictions using XGBoost model
        y_pred_xgb_proba = self.model.predict_proba(data)[:, 1]
        prediction = (y_pred_xgb_proba >= 0.5).astype(int)

        # Calculate SHAP values for the input data
        shap_values = self.explainer.shap_values(data)

        # Generate and save the SHAP plot
        shap_plot_path = self.generate_shap_plot(data, shap_values)

        # Return prediction, probability, and SHAP plot path
        return prediction[0],  shap_plot_path

    def generate_shap_plot(self, data, shap_values):
        # Create the static directory if it doesn't exist
        static_dir = Path('static')
        static_dir.mkdir(parents=True, exist_ok=True)

        # Create SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, data, feature_names=[
            'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Diabetic', 'KidneyDisease'], show=False)

        # Save the SHAP plot to a file
        shap_plot_path = static_dir / "shap_plot.png"
        plt.savefig(shap_plot_path)
        plt.close()

        return shap_plot_path
