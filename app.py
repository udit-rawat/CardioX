import os
import numpy as np
import logging
from flask import Flask, render_template, request, send_from_directory
from src.mlpro.pipeline.prediction import PredictionPipeline

# Set the matplotlib backend to Agg before any other imports
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Path to the static folder for SHAP plots
STATIC_FOLDER = os.path.join(app.root_path, "static")

# Ensure the static folder exists
os.makedirs(STATIC_FOLDER, exist_ok=True)


@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def training():
    try:
        import subprocess
        result = subprocess.run(["python3", "main.py"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("Training successful")
            return "Training Successful!"
        else:
            logging.error(f"Training failed: {result.stderr}")
            return "Training Failed. Check logs for details."
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return "Training Failed. Check logs for details."


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            input_fields = ['Stroke', 'PhysicalHealth', 'MentalHealth',
                            'DiffWalking', 'Diabetic', 'KidneyDisease']
            feature_values = {
                field: request.form[field] for field in input_fields}

            # Convert form data to numpy array for prediction
            data = np.array([float(feature_values[field])
                            for field in input_fields]).reshape(1, -1)

            # Initialize the prediction pipeline and make a prediction
            prediction_pipeline = PredictionPipeline()
            prediction, shap_plot_path = prediction_pipeline.predict_and_explain(
                data)

            # Ensure the plot path is relative to the static folder
            shap_plot_filename = os.path.basename(shap_plot_path)

            logging.info(f"Prediction made: {prediction}")
            logging.info(f"SHAP plot generated: {shap_plot_filename}")

            # Render the results page with prediction, SHAP plot, and features
            return render_template('results.html',
                                   prediction=str(prediction),
                                   shap_plot=shap_plot_filename,
                                   feature_values=feature_values)
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return 'Something went wrong during prediction. Check logs for details.'
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
