import json
import re
import os
import pandas as pd # Added for potential future use with metrics

def parse_log_for_metrics(log_file_path="sleep_model_app.log"):
    """
    Parses the log file to extract model evaluation metrics.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        dict: A dictionary containing evaluation metrics for different models.
              Example: {'random_forest': {'rmse': 1.080, 'r2': 0.060, 'mae': 0.856}, ...}
    """
    metrics = {}
    # Regex to capture model name and metrics from evaluate_models logs
    # Handles global and potentially user-specific evaluations if logged similarly
    metric_pattern = re.compile(
        r"evaluate_models - (\S+?)\s+perf:\s+RMSE=([\d.]+),\s+R²=([-\d.]+),\s+MAE=([\d.]+)"
    )
    # Regex for cross-validation results (if they were logged)
    cv_pattern = re.compile(
         r"(\S+?)\s+CV:\s+RMSE=([\d.]+)±[\d.]+,\s+R²=([-\d.]+)±[\d.]+"
    )


    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found at {log_file_path}. Cannot extract metrics.")
        return metrics

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = metric_pattern.search(line)
                if match:
                    model_name = match.group(1)
                    # Ensure keys are strings and values are floats
                    metrics[model_name] = {
                        'rmse': float(match.group(2)),
                        'r2': float(match.group(3)),
                        'mae': float(match.group(4)),
                        'source': 'evaluation' # Mark as from direct evaluation
                    }
                else:
                    # Check for CV results as a fallback or additional info
                    cv_match = cv_pattern.search(line)
                    if cv_match:
                         model_name_cv = cv_match.group(1)
                         if model_name_cv not in metrics or metrics[model_name_cv].get('source') != 'evaluation':
                            # Prioritize direct evaluation if available, else use CV mean
                            metrics[model_name_cv] = {
                                'rmse': float(cv_match.group(2)), # Mean RMSE from CV
                                'r2': float(cv_match.group(3)),   # Mean R2 from CV
                                'mae': None, # MAE often not in CV summary line like this
                                'source': 'cross-validation (mean)'
                            }

    except Exception as e:
        print(f"Error reading or parsing log file {log_file_path}: {e}")

    return metrics

def interpret_sleep_prediction(response_json, evaluation_metrics=None):
    """
    Interprets the JSON response from the sleep prediction API, optionally
    including evaluation metrics parsed from logs.

    Args:
        response_json (dict): The parsed JSON response from the API.
        evaluation_metrics (dict, optional): Metrics extracted from logs.
                                            Defaults to None.

    Returns:
        str: A human-readable interpretation string.
    """
    output_lines = []

    try:
        prediction = response_json.get("prediction")
        explanation = response_json.get("explanation", {})
        input_features = response_json.get("input_features", {})
        model_used_full_key = explanation.get(
            "model_interpreted",
            response_json.get("model_used_for_prediction_details", "N/A"),
        )
        status = response_json.get("status")

        if status != "success":
            output_lines.append(f"API call failed: {response_json.get('error', 'Unknown error')}")
            return "\n".join(output_lines)

        output_lines.append("--- Sleep Quality Prediction Interpretation ---")
        output_lines.append(f"Predicted Sleep Quality Score: {prediction:.2f} (out of 10)")
        output_lines.append(f"Model Used: {model_used_full_key}")

        # --- Add Evaluation Context ---
        if evaluation_metrics:
            # Try to find metrics for the specific model used (e.g., 'user-specific: User_0_random_forest' -> 'User_0_random_forest')
            # Or for the global part if user-specific wasn't used (e.g., 'global: random_forest' -> 'random_forest')
            model_key_for_metrics = model_used_full_key.split(': ')[-1] # Get 'User_0_random_forest' or 'random_forest'

            if model_key_for_metrics in evaluation_metrics:
                metrics = evaluation_metrics[model_key_for_metrics]
                source = metrics.get('source', 'evaluation')
                output_lines.append(
                    f"  (Context: Model performance on test set [{source}]: "
                    f"RMSE={metrics.get('rmse', 'N/A'):.3f}, "
                    f"R²={metrics.get('r2', 'N/A'):.3f}, "
                    f"MAE={metrics.get('mae', 'N/A')})"
                 )
            elif 'global' in model_used_full_key and model_key_for_metrics in evaluation_metrics:
                 # Fallback for global if specific key not found but base name exists
                 metrics = evaluation_metrics[model_key_for_metrics]
                 source = metrics.get('source', 'evaluation')
                 output_lines.append(
                    f"  (Context: Global model performance on test set [{source}]: "
                    f"RMSE={metrics.get('rmse', 'N/A'):.3f}, "
                    f"R²={metrics.get('r2', 'N/A'):.3f}, "
                    f"MAE={metrics.get('mae', 'N/A')})"
                 )


        output_lines.append("\nInput Values for this Prediction:")
        # Determine the actual features used by the model from SHAP names if possible
        features_in_explanation = explanation.get('shap_feature_names', list(input_features.keys()))
        for feature in features_in_explanation:
             if feature in input_features: # Check if feature exists in input dict
                 output_lines.append(f"  - {feature}: {input_features[feature]}")


        # --- SHAP Interpretation ---
        shap_values = explanation.get("shap_values")
        shap_features = explanation.get("shap_feature_names")
        if shap_values and shap_features and len(shap_values) == len(shap_features):
            shap_contributions = sorted(
                zip(shap_features, shap_values), key=lambda item: abs(item[1]), reverse=True
            )
            output_lines.append("\nKey Factors Influencing Prediction (SHAP Analysis):")
            output_lines.append(" (How much each factor pushed the score away from the baseline)")

            neg_shap_count = 0
            pos_shap_count = 0
            recommendations = {} # Store potential recommendations

            for feature, value in shap_contributions[:5]: # Show top 5
                input_val = input_features.get(feature, "N/A")
                direction = "Increased" if value > 0 else "Decreased"
                impact = abs(value)
                output_lines.append(
                    f"  - {feature} (Input: {input_val}): {direction} score by {impact:.3f}"
                )
                # Generate recommendations based on negative factors
                if value < -0.01 and neg_shap_count < 3: # Threshold for significance
                    recommendations[feature] = (input_val, value)
                    neg_shap_count += 1
                elif value > 0.01 and pos_shap_count < 2:
                     pos_shap_count += 1 # Note positive factors but focus recommendations on negatives

        else:
            output_lines.append("\nSHAP explanation data not available or inconsistent.")
            recommendations = {} # Ensure recommendations dict exists even if SHAP fails

        # --- LIME Interpretation (Optional - can be verbose) ---
        lime_exp = explanation.get("lime_explanation")
        if lime_exp:
            output_lines.append("\nLocal Explanation Details (LIME - Top 3):")
            output_lines.append(" (Factors most influential for *this specific* prediction instance)")
            for feature_cond, weight in lime_exp[:3]:
                 direction = "Increases" if weight > 0 else "Decreases"
                 output_lines.append(f"  - Condition '{feature_cond}' {direction} score by {abs(weight):.3f}")
        # else:
        #     output_lines.append("\nLIME explanation data not available.")


        # --- Generate Recommendations ---
        if recommendations:
            output_lines.append("\nPotential Areas for Improvement (based on SHAP):")
            for feature, (input_val, shap_val) in recommendations.items():
                # Ensure input_val is displayed correctly, even if it's 0 or None
                input_display = input_val if input_val is not None else "N/A"

                rec_text = f"  - {feature} (Input: {input_display}) significantly lowered the predicted score (by {abs(shap_val):.3f}). "
                if feature == "ScreenTime":
                    rec_text += "Consider reducing screen use, especially before bed."
                elif feature == "StressLevel" and shap_val < 0: # Only if SHAP shows it's negative
                    rec_text += "Managing stress through relaxation techniques could help."
                elif feature == "CaffeineIntake":
                    rec_text += "Try reducing caffeine or consuming it earlier in the day."
                elif feature == "BedroomNoise":
                     rec_text += "Ensuring a quieter sleep environment might be beneficial."
                     # Check if input_val is numeric before comparison
                     if isinstance(input_val, (int, float)) and shap_val < 0 and input_val < 0.3:
                         rec_text += " (Note: The model linked lower noise to a lower score here, which is unusual but reflects its learned pattern for you)."
                elif feature == "BedroomLight":
                    rec_text += "Making the bedroom darker could improve sleep."
                elif feature == "EveningAlcohol":
                    rec_text += "Avoiding alcohol close to bedtime often improves sleep quality."
                elif feature == "StepsToday" or feature == "PhysicalActivity":
                     rec_text += "Increasing daily physical activity might lead to better sleep."
                elif feature == "SocialJetlag":
                     rec_text += "Maintaining a more consistent sleep schedule, even on weekends, could help."
                else:
                    rec_text += "Investigating this factor further may be useful."
                output_lines.append(rec_text)
        elif shap_values and shap_features: # Check if SHAP ran but found no strong negatives
             output_lines.append("\nNo strong negative factors identified by SHAP for immediate recommendations.")
        # If SHAP failed, recommendations dict will be empty, so no message is printed here.


        output_lines.append("\n--- End Interpretation ---")

    except Exception as e:
        output_lines.append(f"\nError during interpretation: {e}")
        import traceback
        output_lines.append(traceback.format_exc()) # Add traceback for debugging

    return "\n".join(output_lines)

# --- Main Execution ---
if __name__ == "__main__":
    LOG_FILE = "sleep_model_app.log" # Path to your log file

    # --- Load API Response JSON ---
    # Option 1: Paste the JSON directly (as you provided)
    api_response_data = {
        "explanation": {
            "lime_explanation": [
                ["StepsToday <= -0.46", -0.13613326087323785],
                ["DeepSleepProportion > -0.14", 0.12058740864482971],
                ["BedroomNoise <= -0.27", -0.10926936680735062],
                ["0.03 < StressLevel <= 0.64", 0.07563134094877967],
                ["-0.31 < ScreenTime <= 0.32", -0.06322542715363022],
                ["-0.57 < CircadianStability <= -0.33", 0.037974710681641086],
                ["-0.31 < CaffeineIntake <= 0.65", -0.03334406967269982],
                ["RestingHR <= -0.17", -0.0320641846693986],
                ["-0.51 < PhysicalActivity <= -0.14", -0.024203114457846984],
                ["-1.06 < AvgHRV <= 0.17", -0.02193752805322789],
                ["MindfulnessPractice > -0.49", -0.01470030579248117],
                ["-0.60 < SocialJetlag <= -0.14", 0.011927196281992323],
                ["BedroomLight <= -0.46", 0.009737951158545711],
                ["ExerciseFrequency <= -0.27", 0.007581185221869324],
                ["-0.65 < DietScore <= 0.31", 0.003174446886913849],
                ["-0.53 < EveningAlcohol <= 0.67", 0.0026815957270881854],
            ],
            "model_interpreted": "user-specific: User_0_random_forest",
            "shap_feature_names": [
                "ScreenTime", "StressLevel", "DietScore", "CircadianStability",
                "PhysicalActivity", "CaffeineIntake", "BedroomNoise", "BedroomLight",
                "EveningAlcohol", "ExerciseFrequency", "SocialJetlag", "MindfulnessPractice",
                "AvgHRV", "DeepSleepProportion", "StepsToday", "RestingHR",
            ],
            "shap_values": [
                -0.057062957901507616, 0.023055556463077664, -0.006744444137439132,
                0.03852778129900495, -0.011093658007060489, -0.0319142845304062,
                -0.11791667349946995, -0.052140476958205305, -0.004138887161388993,
                0.026749998874341447, -0.007989283612308403, 0.013844446434328953,
                -0.026747220583880942, -0.01091825015222033, -0.014285188245897492,
                0.02750767464749515,
            ],
        },
        "input_features": {
            # Using the inputs from your example JSON
            "AvgHRV": 60.0,
            "BedroomLight": 0.3,
            "BedroomNoise": 0.2,
            "CaffeineIntake": 100.0,
            "CircadianStability": 0.8,
            "DeepSleepProportion": 0.2,
            "DietScore": 6.0,
            "EveningAlcohol": 1.0,
            "ExerciseFrequency": 3.0,
            "MindfulnessPractice": 1.0,
            "PhysicalActivity": 45.0,
            "RestingHR": 60.0,
            "ScreenTime": 2.5,
            "SocialJetlag": 0.5,
            "StepsToday": 8000.0,
            "StressLevel": 7.0,
            # Add ParticipantID if needed for context, though not used in interpretation logic directly
            "ParticipantID": "User_0"
        },
        "model_used_for_prediction_details": "user-specific: User_0_random_forest",
        "prediction": 4.229714285714287,
        "status": "success",
    }

    # Option 2: Load from a separate JSON file (uncomment to use)
    # api_response_file = "api_output.json"
    # try:
    #     with open(api_response_file, 'r') as f:
    #         api_response_data = json.load(f)
    # except FileNotFoundError:
    #     print(f"Error: API response file not found at {api_response_file}")
    #     api_response_data = None
    # except json.JSONDecodeError:
    #     print(f"Error: Could not decode JSON from {api_response_file}")
    #     api_response_data = None

    # --- Parse Log for Metrics ---
    evaluation_metrics = parse_log_for_metrics(LOG_FILE)
    if evaluation_metrics:
        print(f"Successfully extracted evaluation metrics from {LOG_FILE}:")
        # Optional: Print extracted metrics for verification
        # for model, metrics_data in evaluation_metrics.items():
        #    print(f"  - {model}: {metrics_data}")
    else:
        print(f"Could not extract evaluation metrics from {LOG_FILE}.")


    # --- Generate and Print Interpretation ---
    if api_response_data:
        interpretation_text = interpret_sleep_prediction(api_response_data, evaluation_metrics)
        print("\n" + "="*40 + "\n")
        print(interpretation_text)
    else:
        print("\nCannot generate interpretation as API response data is missing.")
