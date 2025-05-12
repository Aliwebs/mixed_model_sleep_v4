import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

# Import the model class from the other file
from sleep_model import SleepQualityModel

# --- Configuration ---
MODEL_FILE_PATH = "sleep_model_bundle_v2.pkl"
LOG_FILE_NAME = "sleep_model_app.log" # Centralized log file

# --- Configure Logging ---
def setup_logging():
    """Configures root logger for the Flask app."""
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers if necessary (e.g., during reloads)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE_NAME, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=log_date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Log INFO level to console
    console_formatter = logging.Formatter(log_format, datefmt=log_date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger('werkzeug').setLevel(logging.WARNING) # Quieter Flask logs
    logging.info("Logging configured.")

setup_logging()
logger = logging.getLogger(__name__)

# --- Load Model ---
# Load the model once when the application starts
logger.info(f"Attempting to load model from: {MODEL_FILE_PATH}")
try:
    if not os.path.exists(MODEL_FILE_PATH):
        logger.error(f"Model file not found at {MODEL_FILE_PATH}. Cannot start API.")
        # Exit or raise prevents app from starting incorrectly
        raise FileNotFoundError(f"Required model file not found: {MODEL_FILE_PATH}")

    # Use the static method from the class to load
    sleep_model_instance = SleepQualityModel.load_model(file_path=MODEL_FILE_PATH)
    logger.info("Sleep quality model loaded successfully.")

    # Validate essential components after loading
    if not sleep_model_instance.is_scaler_fitted:
         logger.warning("Loaded model's scaler is not fitted. Predictions may fail.")
    if not sleep_model_instance.features_to_scale:
         logger.error("Loaded model has no features defined ('features_to_scale'). Cannot proceed.")
         raise ValueError("Model loaded without defined features.")
    # Add more checks if needed (e.g., at least one model exists)

except Exception as e:
    logger.critical(f"FATAL: Failed to load sleep model: {e}", exc_info=True)
    # Set instance to None to indicate failure
    sleep_model_instance = None
    # Optionally, re-raise to prevent Flask from starting with a broken state
    # raise RuntimeError(f"Failed to initialize model: {e}") from e


# --- Create Flask App ---
app = Flask(__name__)
# Enable CORS for all domains on all routes (adjust in production)
CORS(app, resources={r"/predict": {"origins": "*"}})
logger.info("Flask app created and CORS enabled for /predict.")


# --- Helper Function for Meaningful Response ---
def _generate_meaningful_response(prediction, explanation, input_features):
    """
    Translates raw prediction and explanation data into user-friendly insights.
    """
    insights = {
        "summary": "",
        "key_factors": [],
        "suggestions": [], # General, non-medical suggestions
        "raw_explanation_available": False
    }

    # 1. Summarize Prediction Score
    score = prediction
    if score >= 8.0:
        insights["summary"] = f"Your predicted sleep quality score ({score:.1f}/10) is excellent."
    elif score >= 6.0:
        insights["summary"] = f"Your predicted sleep quality score ({score:.1f}/10) is good."
    elif score >= 4.0:
        insights["summary"] = f"Your predicted sleep quality score ({score:.1f}/10) is average. There might be room for improvement."
    else:
        insights["summary"] = f"Your predicted sleep quality score ({score:.1f}/10) is quite low, suggesting potential issues impacting your sleep."

    # 2. Analyze Explanations (Prioritize LIME if available, fallback to SHAP)
    feature_importance = [] # List of tuples: (feature_name, importance_value, original_value)

    if explanation and 'lime_explanation' in explanation and explanation['lime_explanation']:
        insights["raw_explanation_available"] = True
        lime_list = explanation['lime_explanation'] # List of (feature_str, weight)
        # LIME features might have comparison operators (e.g., 'ScreenTime <= 2.5')
        # We need to map these back to original feature names if possible, or use as is.
        # For simplicity here, we'll parse the feature name part.
        for feature_str, weight in lime_list:
             # Basic parsing: take the part before the first space
             feature_name = feature_str.split(' ')[0]
             if feature_name in input_features:
                 original_value = input_features.get(feature_name, 'N/A')
                 feature_importance.append((feature_name, weight, original_value))
             else:
                 # Handle cases where LIME feature name isn't directly in input_features
                 # Maybe log this or try more complex parsing
                 logger.debug(f"LIME feature '{feature_str}' not directly mapped to input features.")
                 # Use the string as is for now
                 feature_importance.append((feature_str, weight, 'N/A'))

    elif explanation and 'shap_values' in explanation and explanation['shap_values']:
        insights["raw_explanation_available"] = True
        shap_values = explanation['shap_values']
        shap_features = explanation.get('shap_feature_names', [])
        if len(shap_values) == len(shap_features):
            for feature, shap_val in zip(shap_features, shap_values):
                 original_value = input_features.get(feature, 'N/A')
                 feature_importance.append((feature, shap_val, original_value))
        else:
            logger.warning("SHAP values and feature names length mismatch.")

    # 3. Identify Key Factors (Top N positive/negative)
    if feature_importance:
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True) # Sort by absolute importance
        top_n = 3
        positive_factors = []
        negative_factors = []

        for name, importance, value in feature_importance:
            # Use more descriptive names if available
            desc_name = name.replace("_", " ").title() # Simple formatting
            value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)

            if importance > 0 and len(positive_factors) < top_n:
                 positive_factors.append(f"{desc_name} (value: {value_str}) appears to be positively influencing your score.")
            elif importance < 0 and len(negative_factors) < top_n:
                 negative_factors.append(f"{desc_name} (value: {value_str}) seems to be negatively impacting your score.")
            if len(positive_factors) >= top_n and len(negative_factors) >= top_n:
                break # Stop once we have enough factors

        insights["key_factors"] = positive_factors + negative_factors

    # 4. Generate General Suggestions (Example - expand significantly)
    # IMPORTANT: Keep these general and avoid medical advice.
    if feature_importance:
        suggestions_added = set() # Avoid duplicate suggestions
        for name, importance, value in feature_importance[:5]: # Check top 5 factors
            suggestion = None
            if name == "ScreenTime" and importance < 0 and value > 2: # Example threshold
                suggestion = "Consider reducing screen time, especially in the hour before bed."
            elif name == "StressLevel" and importance < 0 and value > 6:
                suggestion = "High stress levels might be affecting your sleep. Exploring relaxation techniques could be beneficial."
            elif name == "PhysicalActivity" and importance > 0 and value < 30:
                 suggestion = "Regular physical activity often helps improve sleep, but avoid intense workouts close to bedtime."
            elif name == "DietScore" and importance < 0 and value < 5:
                 suggestion = "A balanced diet plays a role in sleep. Consider reviewing your evening meals and overall nutrition."
            elif name == "CaffeineIntake" and importance < 0 and value > 100: # Example threshold (mg)
                 suggestion = "High caffeine intake, especially later in the day, can disrupt sleep."
            elif name == "EveningAlcohol" and importance < 0 and value > 0:
                 suggestion = "While alcohol might help fall asleep initially, it often disrupts sleep later in the night."

            if suggestion and suggestion not in suggestions_added:
                insights["suggestions"].append(suggestion)
                suggestions_added.add(suggestion)

    if not insights["key_factors"]:
         insights["key_factors"].append("Could not determine key influencing factors from the explanation.")
    if not insights["suggestions"]:
         insights["suggestions"].append("Maintain consistent sleep routines and a comfortable sleep environment for better sleep quality.")


    return insights


# --- API Routes ---
@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    if sleep_model_instance and sleep_model_instance.is_scaler_fitted:
        return jsonify({"status": "ok", "message": "Sleep model service is running."}), 200
    else:
        logger.error("Health check failed: Model not loaded or scaler not fitted.")
        return jsonify({"status": "error", "message": "Sleep model service is not healthy (model loading issue)."}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handles prediction requests. Expects JSON payload with 'instance'
    and optional 'user_id' and 'model_type'.
    """
    endpoint_logger = logging.getLogger(f"{__name__}.predict_route")
    endpoint_logger.info("Received request on /predict")

    # Check if model loaded correctly during startup
    if sleep_model_instance is None:
        endpoint_logger.error("Prediction failed because the model was not loaded.")
        return jsonify({"error": "Model not initialized. Cannot process request.", "status": "error"}), 500

    # --- Input Validation ---
    if not request.is_json:
        endpoint_logger.warning("Request is not JSON.")
        return jsonify({"error": "Request must be JSON.", "status": "error"}), 400

    data_json = request.json
    instance_dict = data_json.get("instance")
    if not instance_dict or not isinstance(instance_dict, dict):
        endpoint_logger.warning("Missing or invalid 'instance' dictionary in payload.")
        return jsonify({"error": "Missing 'instance' dictionary in payload.", "status": "error"}), 400

    # Optional parameters
    user_id = data_json.get("user_id") # Can be None
    model_type_pref = data_json.get("model_type", "random_forest") # Default model type

    endpoint_logger.info(f"Processing prediction for User: {user_id if user_id else 'N/A'}, Model Type Pref: {model_type_pref}")
    endpoint_logger.debug(f"Input instance data: {instance_dict}")

    # --- Prediction and Interpretation ---
    try:
        # Ensure all features the model expects are present, default if necessary
        # This should ideally align with how preprocess_data handles missing features
        processed_input_dict = instance_dict.copy()
        for f_name in sleep_model_instance.features_to_scale:
            if f_name not in processed_input_dict:
                endpoint_logger.warning(f"Feature '{f_name}' missing in API input, defaulting to 0 for prediction/interpretation.")
                processed_input_dict[f_name] = 0 # Or np.nan - must match preprocessing logic

        # 1. Predict
        prediction_value = sleep_model_instance.predict_sleep_quality(
            processed_input_dict, model_type=model_type_pref, user_id=user_id
        )
        endpoint_logger.info(f"Prediction successful: {prediction_value}")

        # 2. Interpret
        explanation_output = sleep_model_instance.interpret_model(
            processed_input_dict, model_type=model_type_pref, user_id=user_id
        )
        endpoint_logger.info("Interpretation successful.")
        endpoint_logger.debug(f"Raw explanation output: {explanation_output}")

        # 3. Generate Meaningful Response
        meaningful_response = _generate_meaningful_response(
            prediction_value, explanation_output, processed_input_dict
        )
        endpoint_logger.info("Meaningful response generated.")

        # --- Format and Return Response ---
        response_payload = {
            "status": "success",
            "prediction": prediction_value,
            "insights": meaningful_response, # User-friendly insights
            "model_details": { # Info about the model used
                 "type_requested": model_type_pref,
                 "user_id_provided": user_id,
                 "model_used": explanation_output.get('model_interpreted', 'N/A')
            },
            # Optionally include raw explanations if needed by frontend
            # "raw_explanation": explanation_output
        }
        return jsonify(response_payload), 200

    except (ValueError, TypeError) as e:
        endpoint_logger.error(f"Input validation or prediction/interpretation error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Bad Request: {str(e)}", "status": "error"}), 400
    except RuntimeError as e:
         endpoint_logger.error(f"Runtime error during processing (e.g., scaler issue): {str(e)}", exc_info=True)
         return jsonify({"error": f"Internal Server Error: {str(e)}", "status": "error"}), 500
    except Exception as e:
        endpoint_logger.critical(f"Unexpected internal server error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred.", "status": "error"}), 500


# --- Run Flask App (for local development) ---
# This part is typically NOT used when deploying with Gunicorn/WSGI
if __name__ == "__main__":
    logger.info("Starting Flask development server...")
    # Make sure model loaded before running
    if sleep_model_instance is None:
         logger.critical("Cannot start development server: Model failed to load.")
    else:
        # Use host='0.0.0.0' to make accessible on network
        # debug=False is safer, debug=True enables auto-reload but is insecure
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)), debug=False)

# --- End of app.py ---
