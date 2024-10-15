import pandas as pd
from src.mlpro.pipeline.prediction import PredictionPipeline


def test_prediction_pipeline():
    # Create sample input data
    sample_data = pd.DataFrame({
        'Stroke': [1],
        'PhysicalHealth': [10],
        'MentalHealth': [5],
        'DiffWalking': [1],
        'Diabetic': [0],
        'KidneyDisease': [0]
    })
    sample_data = sample_data.to_numpy()

    # Initialize the PredictionPipeline
    pipeline = PredictionPipeline()

    # Make predictions and get SHAP plot path
    prediction, shap_plot_path = pipeline.predict_and_explain(sample_data)

    # Output results
    print(f"Prediction: {prediction}")
    print(f"SHAP plot saved at: {shap_plot_path}")


if __name__ == "__main__":
    test_prediction_pipeline()
