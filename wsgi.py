import os
import logging
from mixed_model_sleep_analysis_v4 import SleepQualityModel # Assuming your main script is named sleep_model_v2.py

# Configure basic logging for the WSGI entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define the path to your saved model file.
# On a deployment platform, this file needs to be included in your uploaded code.
MODEL_FILE_PATH = "sleep_model_bundle_v2.pkl"

# --- Load the Model ---
# This code runs once when the Gunicorn worker starts
logger.info(f"Loading sleep model from {MODEL_FILE_PATH}...")
try:
    # Make sure SleepQualityModel.load_model is a static method or class method
    loaded_model_instance = SleepQualityModel.load_model(file_path=MODEL_FILE_PATH)
    logger.info("Sleep model loaded successfully.")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_FILE_PATH}. Ensure it's included in deployment.", exc_info=True)
    # Depending on your error handling, you might want to exit or indicate failure
    # For now, the loaded_model_instance will be None, leading to errors if used.
    loaded_model_instance = None # Explicitly set to None on error
except Exception as e:
    logger.error(f"Error loading sleep model: {str(e)}", exc_info=True)
    loaded_model_instance = None


# --- Create the Flask Application Instance ---
# This also runs when the Gunicorn worker starts
if loaded_model_instance:
    # Call the create_api method from your loaded model instance
    # Assuming create_api returns the Flask app object
    app = loaded_model_instance.create_api()
    logger.info("Flask app created from loaded model instance.")
else:
    logger.error("Could not create Flask app because model loading failed.")
    # Create a dummy Flask app that returns an error for all requests
    from flask import Flask, jsonify
    app = Flask(__name__)
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def health_check_or_error(path):
        return jsonify({"status": "error", "message": "Backend initialization failed: Model could not be loaded."}), 500


#

