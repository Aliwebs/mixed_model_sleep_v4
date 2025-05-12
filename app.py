import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re # For parsing feature names

# Import the model class from the other file
# Ensure sleep_model.py is in the same directory or Python path
try:
    from sleep_model import SleepQualityModel
except ImportError:
    logging.critical("Failed to import SleepQualityModel from sleep_model.py. Ensure the file exists and is accessible.")
    # Define a dummy class if import fails to prevent immediate crash,
    # but the app won't function correctly.
    class SleepQualityModel:
        def __init__(self, *args, **kwargs): pass
        @staticmethod
        def load_model(*args, **kwargs): return None # Simulate load failure

# --- Configuration ---
MODEL_FILE_PATH = os.environ.get("MODEL_FILE_PATH", "sleep_model_bundle_v2.pkl")
LOG_FILE_NAME = "sleep_model_app.log" # Centralized log file
FLASK_PORT = int(os.environ.get("PORT", 5001))
FLASK_HOST = os.environ.get("HOST", "0.0.0.0")
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

# --- Configure Logging ---
def setup_logging():
    """Configures root logger for the Flask app."""
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"
    root_logger = logging.getLogger()
    log_level = logging.DEBUG if FLASK_DEBUG else logging.INFO
    root_logger.setLevel(log_level)

    # Remove existing handlers if necessary (e.g., during reloads)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE_NAME, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, datefmt=log_date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Failed to configure file logging: {e}")


    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, datefmt=log_date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger('werkzeug').setLevel(logging.WARNING) # Quieter Flask logs
    logging.info(f"Logging configured at level {logging.getLevelName(log_level)}.")

setup_logging()
logger = logging.getLogger(__name__)

# --- Load Model ---
# Load the model once when the application starts
logger.info(f"Attempting to load model from: {MODEL_FILE_PATH}")
sleep_model_instance = None # Initialize as None
try:
    if not os.path.exists(MODEL_FILE_PATH):
        logger.error(f"Model file not found at {MODEL_FILE_PATH}. Cannot start API.")
        # Exit or raise prevents app from starting incorrectly
        raise FileNotFoundError(f"Required model file not found: {MODEL_FILE_PATH}")

    # Use the static method from the class to load
    sleep_model_instance = SleepQualityModel.load_model(file_path=MODEL_FILE_PATH)

    if sleep_model_instance is None:
         # This case should ideally be caught by exceptions in load_model
         raise ValueError("SleepQualityModel.load_model returned None.")

    logger.info("Sleep quality model loaded successfully.")

    # Validate essential components after loading
    if not hasattr(sleep_model_instance, 'is_scaler_fitted') or not sleep_model_instance.is_scaler_fitted:
         logger.warning("Loaded model's scaler is not fitted or attribute missing. Predictions may fail.")
    if not hasattr(sleep_model_instance, 'features_to_scale') or not sleep_model_instance.features_to_scale:
         logger.error("Loaded model has no features defined ('features_to_scale'). Cannot proceed.")
         raise ValueError("Model loaded without defined features.")
    # Add more checks if needed (e.g., at least one model exists)

except Exception as e:
    logger.critical(f"FATAL: Failed to load sleep model: {e}", exc_info=True)
    # Keep instance as None to indicate failure
    sleep_model_instance = None
    # Optionally, re-raise to prevent Flask from starting with a broken state
    # raise RuntimeError(f"Failed to initialize model: {e}") from e


# --- Create Flask App ---
app = Flask(__name__)
# Enable CORS for all domains on all routes (adjust in production)
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow all origins for /predict and /health
logger.info("Flask app created and CORS enabled for all routes.")


# --- Helper Function for Parsing and Formatting Feature Names ---
def _get_simple_feature_name(complex_name, known_features):
    """
    Extracts a simpler, core feature name from potentially complex explainer output.
    Tries to match against known features.
    """
    if not isinstance(complex_name, str): # Handle non-string inputs gracefully
        return "Unknown Factor"

    # First, check if the complex_name itself is a known feature (case-insensitive check)
    for kf in known_features:
        if kf.lower() == complex_name.lower():
            return kf # Return the known feature name with original casing

    # If LIME-style output (e.g., "Feature < Value" or "Value < Feature <= Value2")
    # Try to extract a word that matches one of the known features
    words_in_complex_name = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', complex_name)
    for word in words_in_complex_name:
        for kf in known_features:
            if kf.lower() == word.lower():
                return kf # Found a known feature within the complex string

    # Fallback: if no known feature found, take the first sensible part
    if words_in_complex_name:
        # Avoid returning numbers or single letters if they appear first
        for word in words_in_complex_name:
            if not word.isdigit() and len(word) > 1:
                return word
        # If only numbers/single letters found, return the first one anyway
        return words_in_complex_name[0]

    return complex_name # Absolute fallback

def _format_feature_name_for_display(feature_name):
    """Converts a feature_name like 'ScreenTime' to 'Screen Time' or 'AvgHRV' to 'Average HRV'."""
    if not feature_name or not isinstance(feature_name, str):
        return "This factor"

    # Specific common acronyms or terms (case-insensitive matching)
    replacements = {
        "avghrv": "Average HRV",
        "restinghr": "Resting Heart Rate",
        "stepstoday": "Steps Taken Today",
        "deepsleepproportion": "Deep Sleep Percentage",
        "screentime": "Screen Time",
        "stresslevel": "Stress Level",
        "dietscore": "Diet Score",
        "circadianstability": "Circadian Stability",
        "physicalactivity": "Physical Activity",
        "caffeineintake": "Caffeine Intake",
        "bedroomnoise": "Bedroom Noise",
        "bedroomlight": "Bedroom Light",
        "eveningalcohol": "Evening Alcohol",
        "exercisefrequency": "Exercise Frequency",
        "socialjetlag": "Social Jetlag",
        "mindfulnesspractice": "Mindfulness Practice"
    }
    feature_name_lower = feature_name.lower()
    if feature_name_lower in replacements:
        return replacements[feature_name_lower]

    # General case: add space before capitals (camelCase to Title Case)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', feature_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    # Capitalize first letter and handle underscores
    return s2.replace("_", " ").strip().capitalize()


# --- Helper Function for Meaningful Response (UPDATED) ---
def _generate_meaningful_response(prediction, explanation, input_features, known_feature_list):
    """
    Translates raw prediction and explanation data into user-friendly insights.
    """
    insights = {
        "summary": "",
        "description_for_average_user": [], # New section
        "key_technical_factors": [],      # Renamed from key_factors
        "suggestions": [],
        "raw_explanation_available": False
    }

    # 1. Summarize Prediction Score
    score = prediction
    if not isinstance(score, (int, float)):
        insights["summary"] = "Could not determine prediction score."
        logger.warning(f"Invalid prediction score type: {type(score)}")
        score = 0 # Default for logic below
    else:
        if score >= 8.0:
            insights["summary"] = f"Your predicted sleep quality score is {score:.1f}/10. That's excellent!"
        elif score >= 6.0:
            insights["summary"] = f"Your predicted sleep quality score is {score:.1f}/10. That's pretty good."
        elif score >= 4.0:
            insights["summary"] = f"Your predicted sleep quality score is {score:.1f}/10. This is about average, and there might be some things you can look into for improvement."
        else:
            insights["summary"] = f"Your predicted sleep quality score is {score:.1f}/10. This is on the lower side, suggesting some factors might be significantly impacting your sleep."

    # 2. Analyze Explanations
    feature_importance_data = [] # List of tuples: (raw_name, simple_name, importance_value, original_value_str)
    has_lime = explanation and 'lime_explanation' in explanation and explanation['lime_explanation']
    has_shap = explanation and 'shap_values' in explanation and explanation['shap_values']

    if has_lime:
        insights["raw_explanation_available"] = True
        lime_list = explanation['lime_explanation'] # List of (feature_str, weight)
        for feature_str, weight in lime_list:
            simple_name = _get_simple_feature_name(feature_str, known_feature_list)
            display_name = _format_feature_name_for_display(simple_name)
            original_value = input_features.get(simple_name, 'N/A') # Get original value using simple_name
            value_str = f"{original_value:.2f}" if isinstance(original_value, (int, float)) else str(original_value)
            feature_importance_data.append((feature_str, display_name, weight, value_str))
        logger.debug("Processed LIME explanation for insights.")

    elif has_shap:
        insights["raw_explanation_available"] = True
        shap_values_list = explanation['shap_values']
        shap_features = explanation.get('shap_feature_names', [])
        if len(shap_values_list) == len(shap_features):
            for feature_name_raw, shap_val in zip(shap_features, shap_values_list):
                simple_name = _get_simple_feature_name(feature_name_raw, known_feature_list) # Should be same as raw for SHAP
                display_name = _format_feature_name_for_display(simple_name)
                original_value = input_features.get(simple_name, 'N/A')
                value_str = f"{original_value:.2f}" if isinstance(original_value, (int, float)) else str(original_value)
                feature_importance_data.append((feature_name_raw, display_name, shap_val, value_str))
            logger.debug("Processed SHAP explanation for insights.")
        else:
            logger.warning("SHAP values and feature names length mismatch during insight generation.")
    else:
        logger.warning("No LIME or SHAP explanation data found to generate insights.")


    # 3. Identify and Describe Key Factors
    if feature_importance_data:
        # Sort by absolute importance for picking top N
        try:
            feature_importance_data.sort(key=lambda x: abs(x[2]), reverse=True)
        except TypeError as sort_e:
             logger.error(f"Error sorting feature importance data (mixed types?): {sort_e}. Data: {feature_importance_data}")
             # Attempt to filter out non-numeric importance values if possible
             feature_importance_data = [item for item in feature_importance_data if isinstance(item[2], (int, float))]
             if feature_importance_data:
                 feature_importance_data.sort(key=lambda x: abs(x[2]), reverse=True)
             else:
                 logger.error("Could not recover numeric importance data for sorting.")


        top_n_display = min(len(feature_importance_data), 5) # Show up to 5 factors

        for raw_name, display_name, importance, value_str in feature_importance_data[:top_n_display]:
            # Ensure importance is numeric before proceeding
            if not isinstance(importance, (int, float)):
                logger.warning(f"Skipping factor '{raw_name}' due to non-numeric importance: {importance}")
                continue

            # Populate Key Technical Factors (more direct from explainer)
            technical_desc = f"{raw_name} (your input: {value_str})"
            if importance > 0.01: # Using a small threshold to avoid "0.00 influence"
                insights["key_technical_factors"].append(f"{technical_desc} appears to be positively influencing your score (importance: {importance:.2f}).")
            elif importance < -0.01:
                insights["key_technical_factors"].append(f"{technical_desc} seems to be negatively impacting your score (importance: {importance:.2f}).")
            else:
                insights["key_technical_factors"].append(f"{technical_desc} has a minor or negligible influence on your score (importance: {importance:.2f}).")

            # Populate Description for Average User (simpler language)
            user_friendly_sentence = ""
            if value_str != 'N/A':
                value_context = f" (your current input is {value_str})"
            else:
                value_context = "" # Avoid showing N/A to user

            # Example user-friendly sentences (expand these)
            if importance > 0.01:
                if display_name.lower() == "steps taken today" and isinstance(input_features.get("StepsToday"), (int, float)) and input_features.get("StepsToday", 0) > 7000 :
                     user_friendly_sentence = f"Great job on your {display_name.lower()}! Getting plenty of steps{value_context} is likely helping your sleep quality."
                elif display_name.lower() == "physical activity" and isinstance(input_features.get("PhysicalActivity"), (int, float)) and input_features.get("PhysicalActivity", 0) > 30 :
                     user_friendly_sentence = f"Your level of {display_name.lower()}{value_context} seems beneficial for your sleep."
                else:
                     user_friendly_sentence = f"{display_name}{value_context} seems to be having a positive effect on your sleep quality."
            elif importance < -0.01:
                if display_name.lower() == "screen time" and isinstance(input_features.get("ScreenTime"), (int, float)) and input_features.get("ScreenTime", 0) > 2:
                    user_friendly_sentence = f"Your {display_name.lower()}{value_context} might be making it a bit harder to get good sleep, especially if it's close to bedtime."
                elif display_name.lower() == "stress level" and isinstance(input_features.get("StressLevel"), (int, float)) and input_features.get("StressLevel", 0) > 6:
                    user_friendly_sentence = f"High {display_name.lower()}{value_context} can often impact sleep. This might be an area to focus on."
                elif display_name.lower() == "caffeine intake" and isinstance(input_features.get("CaffeineIntake"), (int, float)) and input_features.get("CaffeineIntake", 0) > 100:
                    user_friendly_sentence = f"The amount of {display_name.lower()}{value_context} could be disrupting your sleep, especially if consumed later in the day."
                else:
                    user_friendly_sentence = f"{display_name}{value_context} could be negatively affecting your sleep. It might be worth looking into this."
            else: # Minor influence
                user_friendly_sentence = f"{display_name}{value_context} doesn't seem to be a major factor for your sleep quality right now."

            if user_friendly_sentence:
                insights["description_for_average_user"].append(user_friendly_sentence)

    # 4. Generate General Suggestions (can be refined based on the new descriptions)
    suggestions_added = set()
    for raw_name, display_name, importance, value_str in feature_importance_data[:top_n_display]:
        # Ensure importance is numeric
        if not isinstance(importance, (int, float)):
            continue

        suggestion = None
        # Get original numeric value for threshold checks if possible
        simple_name_for_value = _get_simple_feature_name(raw_name, known_feature_list)
        original_value_for_suggestion = input_features.get(simple_name_for_value)

        # Example suggestions (expand these)
        if display_name == "Screen Time" and importance < -0.01 and isinstance(original_value_for_suggestion, (int,float)) and original_value_for_suggestion > 2:
            suggestion = "Consider reducing screen time, especially in the hour before bed, as it might be affecting your sleep."
        elif display_name == "Stress Level" and importance < -0.01 and isinstance(original_value_for_suggestion, (int,float)) and original_value_for_suggestion > 6:
            suggestion = "High stress levels can impact sleep. Exploring relaxation techniques (like mindfulness or deep breathing) could be beneficial."
        elif display_name == "Caffeine Intake" and importance < -0.01 and isinstance(original_value_for_suggestion, (int,float)) and original_value_for_suggestion > 100 : # e.g. 100mg
            suggestion = "High caffeine intake, especially later in the day, can disrupt sleep. You might want to see if reducing it or having it earlier helps."
        elif display_name == "Evening Alcohol" and importance < -0.01 and isinstance(original_value_for_suggestion, (int,float)) and original_value_for_suggestion > 0:
            suggestion = "While alcohol might seem to help you fall asleep, it can often disrupt sleep quality later in the night. Consider avoiding it close to bedtime."
        elif display_name == "Physical Activity" and importance > 0.01 and isinstance(original_value_for_suggestion, (int,float)) and original_value_for_suggestion < 30:
             suggestion = "Regular physical activity often helps improve sleep. Even moderate activity earlier in the day can make a difference, but avoid intense workouts close to bedtime."

        if suggestion and suggestion not in suggestions_added:
            insights["suggestions"].append(suggestion)
            suggestions_added.add(suggestion)

    # Default messages if no specific factors/suggestions identified
    if not insights["description_for_average_user"]:
         insights["description_for_average_user"].append("We couldn't pinpoint specific major factors from your current data, but general sleep hygiene is always important.")
    if not insights["key_technical_factors"]:
         insights["key_technical_factors"].append("Could not determine key influencing factors from the explanation model.")
    if not insights["suggestions"]:
         insights["suggestions"].append("For better sleep, try to maintain a consistent sleep schedule, ensure your bedroom is dark, quiet, and cool, and create a relaxing bedtime routine.")

    return insights


# --- API Routes ---
@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    if sleep_model_instance and hasattr(sleep_model_instance, 'is_scaler_fitted') and sleep_model_instance.is_scaler_fitted:
        # Basic check: model loaded and scaler seems fitted
        return jsonify({"status": "ok", "message": "Sleep model service is running."}), 200
    elif sleep_model_instance:
        logger.warning("Health check warning: Model loaded but scaler not fitted or attribute missing.")
        return jsonify({"status": "warning", "message": "Sleep model service is running but may have issues (scaler not fitted)."}), 200
    else:
        logger.error("Health check failed: Model instance is None (not loaded).")
        return jsonify({"status": "error", "message": "Sleep model service is not healthy (model not loaded)."}), 500


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
        return jsonify({"error": "Request must be JSON.", "status": "error"}), 415 # Use 415 Unsupported Media Type

    data_json = request.json
    if not data_json: # Handle empty JSON body
         endpoint_logger.warning("Received empty JSON payload.")
         return jsonify({"error": "Request body cannot be empty.", "status": "error"}), 400

    instance_dict = data_json.get("instance")
    if not instance_dict or not isinstance(instance_dict, dict):
        endpoint_logger.warning("Missing or invalid 'instance' dictionary in payload.")
        return jsonify({"error": "Missing 'instance' dictionary in payload.", "status": "error"}), 400

    # Optional parameters
    user_id = data_json.get("user_id") # Can be None
    model_type_pref = data_json.get("model_type", "random_forest") # Default model type

    endpoint_logger.info(f"Processing prediction for User: {user_id if user_id else 'N/A'}, Model Type Pref: {model_type_pref}")
    endpoint_logger.debug(f"Input instance data (first few keys): {dict(list(instance_dict.items())[:5])}") # Log snippet

    # --- Prediction and Interpretation ---
    try:
        # Ensure sleep_model_instance and its features_to_scale are available
        if not hasattr(sleep_model_instance, 'features_to_scale') or not sleep_model_instance.features_to_scale:
            endpoint_logger.error("Model instance or features_to_scale not available.")
            return jsonify({"error": "Model not properly initialized (missing features).", "status": "error"}), 500

        # Prepare input dict, ensuring all expected features are present
        processed_input_dict = instance_dict.copy()
        missing_features = []
        for f_name in sleep_model_instance.features_to_scale:
            if f_name not in processed_input_dict:
                missing_features.append(f_name)
                # Defaulting to 0, align with preprocessing logic if different
                processed_input_dict[f_name] = 0
        if missing_features:
             endpoint_logger.warning(f"Features missing in API input, defaulted to 0: {missing_features}")


        # 1. Predict
        # Pass the dict with potentially added defaults
        prediction_value = sleep_model_instance.predict_sleep_quality(
            processed_input_dict, model_type=model_type_pref, user_id=user_id
        )
        endpoint_logger.info(f"Prediction successful: {prediction_value}")

        # 2. Interpret
        # Pass the same dict used for prediction
        explanation_output = sleep_model_instance.interpret_model(
            processed_input_dict, model_type=model_type_pref, user_id=user_id
        )
        endpoint_logger.info("Interpretation successful.")
        endpoint_logger.debug(f"Raw explanation output keys: {list(explanation_output.keys()) if isinstance(explanation_output, dict) else 'N/A'}")

        # 3. Generate Meaningful Response
        # Pass the list of known features for better name parsing
        meaningful_response = _generate_meaningful_response(
            prediction_value,
            explanation_output,
            processed_input_dict, # Pass the potentially modified dict
            sleep_model_instance.features_to_scale # Pass known features
        )
        endpoint_logger.info("Meaningful response generated.")

        # --- Format and Return Response ---
        response_payload = {
            "status": "success",
            "prediction": prediction_value, # This is the float value
            "insights": meaningful_response,
            "model_details": {
                 "type_requested": model_type_pref,
                 "user_id_provided": user_id,
                 "model_used": explanation_output.get('model_interpreted', 'N/A') if isinstance(explanation_output, dict) else 'N/A'
            },
            # Optionally include raw explanations if needed by frontend, check for errors first
            # "raw_explanation_data": explanation_output if isinstance(explanation_output, dict) and 'error' not in explanation_output else None
        }
        return jsonify(response_payload), 200

    except (ValueError, TypeError) as e:
        endpoint_logger.error(f"Input validation or prediction/interpretation error: {str(e)}", exc_info=True)
        # Provide more specific error message if possible
        error_msg = f"Bad Request: {str(e)}"
        if "scaler is not fitted" in str(e).lower():
             error_msg = "Internal Server Error: Model scaler is not ready."
             return jsonify({"error": error_msg, "status": "error"}), 500
        return jsonify({"error": error_msg, "status": "error"}), 400
    except RuntimeError as e:
         endpoint_logger.error(f"Runtime error during processing (e.g., scaler issue): {str(e)}", exc_info=True)
         return jsonify({"error": f"Internal Server Error: {str(e)}", "status": "error"}), 500
    except FileNotFoundError as e: # Should be caught at startup, but as fallback
        endpoint_logger.critical(f"Model file not found during request processing: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error: Model file missing.", "status": "error"}), 500
    except Exception as e:
        endpoint_logger.critical(f"Unexpected internal server error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred on the server.", "status": "error"}), 500


# --- Run Flask App (for local development) ---
# This part is typically NOT used when deploying with Gunicorn/WSGI
if __name__ == "__main__":
    logger.info("Starting Flask development server...")
    # Make sure model loaded before running
    if sleep_model_instance is None:
         logger.critical("Cannot start development server: Model failed to load.")
    else:
        # Use host/port/debug settings from config/env vars
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

# --- End of app.py ---
