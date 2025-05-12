import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt # Only if visualization is needed here

# Import the model class
from sleep_model import SleepQualityModel

# --- Configuration ---
DATA_FILE_PATH = "synthetic_sleep_enhanced_v3.csv"
MODEL_SAVE_PATH = "sleep_model_bundle_v2.pkl"
LOG_FILE_NAME = "sleep_model_training.log" # Separate log for training

# --- Configure Logging ---
def setup_training_logging():
    """Configures logging specifically for the training script."""
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"
    root_logger = logging.getLogger() # Get root logger
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler for training logs
    file_handler = RotatingFileHandler(
        LOG_FILE_NAME, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=log_date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, datefmt=log_date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.info("Training script logging configured.")

setup_training_logging()
logger = logging.getLogger(__name__)


# --- Data Generation (Copied from original main) ---
def generate_synthetic_data_if_needed(file_path=DATA_FILE_PATH, num_rows=1000, num_participants=50):
    if os.path.exists(file_path):
        logger.info(f"Data file {file_path} already exists. Skipping generation.")
        return

    logger.info(f"Generating synthetic data ({num_rows} rows, {num_participants} participants) at {file_path}...")
    np.random.seed(42)
    data = pd.DataFrame()
    data['ParticipantID'] = np.random.choice([f'User_{i}' for i in range(num_participants)], num_rows)
    base_date = datetime(2024, 1, 1)
    data['Date'] = [base_date + timedelta(days=int(i)) for i in np.random.randint(0, 365, num_rows)]
    data.sort_values(by=['ParticipantID', 'Date'], inplace=True)

    # Lifestyle & Environmental Factors
    data['ScreenTime'] = np.random.uniform(0, 6, num_rows)
    data['StressLevel'] = np.random.randint(1, 11, num_rows)
    data['DietScore'] = np.random.randint(1, 11, num_rows)
    data['PhysicalActivity'] = np.random.uniform(0, 120, num_rows)
    data['CaffeineIntake'] = np.random.choice([0, 50, 100, 150, 200], num_rows, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    data['BedroomNoise'] = np.random.uniform(0, 1, num_rows)
    data['BedroomLight'] = np.random.uniform(0, 1, num_rows)
    data['EveningAlcohol'] = np.random.choice([0, 1, 2, 3], num_rows, p=[0.7, 0.15, 0.1, 0.05])
    data['ExerciseFrequency'] = np.random.randint(0, 8, num_rows)
    data['MindfulnessPractice'] = np.random.choice([0, 1], num_rows, p=[0.8, 0.2])

    # Derived & Stability Metrics
    data['CircadianStability'] = 1 - np.abs(np.random.normal(0, 0.2, num_rows))
    data['SocialJetlag'] = np.abs(np.random.normal(0, 1, num_rows))

    # Placeholder for Wearable Data
    data['AvgHRV'] = np.random.uniform(20, 100, num_rows)
    data['DeepSleepProportion'] = np.random.uniform(0.05, 0.35, num_rows)
    data['StepsToday'] = np.random.randint(500, 20000, num_rows)
    data['RestingHR'] = np.random.randint(45, 90, num_rows)

    # Target: SleepQuality (1-10)
    quality_base = 5
    quality_base -= 0.3 * data['ScreenTime'] / 6
    quality_base -= 0.5 * data['StressLevel'] / 10
    quality_base += 0.3 * data['DietScore'] / 10
    quality_base -= 0.2 * data['BedroomNoise']
    quality_base -= 0.2 * data['BedroomLight']
    quality_base += 0.2 * data['PhysicalActivity'] / 120
    quality_base -= 0.3 * (data['CaffeineIntake'] > 100)
    quality_base -= 0.4 * (data['EveningAlcohol'] > 0)
    quality_base += 0.3 * data['MindfulnessPractice']
    quality_base += 0.2 * data['DeepSleepProportion'] / 0.35
    quality_base -= 0.1 * data['SocialJetlag'] / 2
    quality_base += 0.2 * data['CircadianStability']
    data['SleepQuality'] = np.clip(quality_base + np.random.normal(0, 1.0, num_rows), 1, 10).round(1)

    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Synthetic data generated and saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save synthetic data to {file_path}: {e}")
        raise


# --- Main Training and Evaluation Workflow ---
def run_training_pipeline():
    """Executes the model training, evaluation, and saving process."""
    logger.info("--- Starting Sleep Model Training Pipeline ---")
    try:
        # 1. Generate/Load Data
        generate_synthetic_data_if_needed(DATA_FILE_PATH)
        sleep_model = SleepQualityModel(random_state=42)
        full_data = sleep_model.load_data(DATA_FILE_PATH)

        # 2. Split Data (before fitting scaler)
        # Using participant-based split is generally good practice
        train_df_raw, test_df_raw = sleep_model.split_data(full_data, test_size=0.25, temporal=False)

        if train_df_raw.empty:
            logger.error("Training data is empty after split. Aborting.")
            return
        if test_df_raw.empty:
            logger.warning("Test data is empty after split. Evaluation will be skipped.")

        # 3. Fit Scaler (ONLY on training data)
        sleep_model.fit_scaler(train_df_raw) # Fit on raw training data

        # 4. Preprocess Data (using the fitted scaler)
        train_df_processed = sleep_model.preprocess_data(train_df_raw, is_training_data=True)
        if not test_df_raw.empty:
            test_df_processed = sleep_model.preprocess_data(test_df_raw, is_training_data=False) # Use False for test/new data
        else:
            test_df_processed = pd.DataFrame() # Empty dataframe

        # --- Optional: EDA and Visualization on Processed Training Data ---
        # logger.info("\n--- Performing EDA on Processed Training Data ---")
        # eda_results = sleep_model.exploratory_data_analysis(train_df_processed.copy())
        # logger.info(f"EDA Correlations with Target:\n{eda_results.get('correlations', 'N/A')}")
        # sleep_model.visualize_data(train_df_processed.copy())
        # # If saving plots: plt.savefig("eda_plot.png"); plt.close()

        # 5. Fit Global Models (on processed training data)
        sleep_model.fit_global_models(train_df_processed.copy())

        # 6. Evaluate Global Models (on processed test data)
        if not test_df_processed.empty:
            logger.info("\n--- Evaluating Global Models on Test Set ---")
            global_eval_results = sleep_model.evaluate_models(test_df_processed.copy())
            logger.info(f"Global Model Evaluation Results:\n{global_eval_results}")
        else:
            logger.info("Skipping global model evaluation as test set is empty.")

        # 7. Cross-Validate Global Models (on processed training data)
        logger.info("\n--- Cross-Validating Global Models on Training Set ---")
        # Use train_df_processed for CV to avoid data leakage from test set
        cv_results = sleep_model.cross_validate_global_models(train_df_processed.copy(), n_splits=3)
        logger.info(f"Global Model Cross-Validation Results:\n{cv_results}")

        # 8. Simulate User-Specific Fine-Tuning (using processed training data)
        logger.info("\n--- Simulating User-Specific Fine-Tuning ---")
        unique_users_in_train = train_df_processed[sleep_model.participant_id_col].unique()
        # Fine-tune for a subset of users (e.g., first 2 for demo)
        users_to_fine_tune = unique_users_in_train[:2]

        for user_id_to_tune in users_to_fine_tune:
            # Get this user's processed training data
            user_specific_training_data = train_df_processed[
                train_df_processed[sleep_model.participant_id_col] == user_id_to_tune
            ]
            if not user_specific_training_data.empty:
                tuned = sleep_model.fine_tune_model_for_user(
                    user_id=user_id_to_tune,
                    user_df=user_specific_training_data.copy(),
                    model_type="random_forest", # Or try 'xgboost'
                    min_samples=5 # Lower min_samples for demo
                )
                if tuned and not test_df_processed.empty:
                    # Evaluate this user's model on their portion of the processed test set
                    user_test_data = test_df_processed[test_df_processed[sleep_model.participant_id_col] == user_id_to_tune]
                    if not user_test_data.empty:
                        logger.info(f"\n--- Evaluating Fine-Tuned Model for User: {user_id_to_tune} ---")
                        user_eval = sleep_model.evaluate_models(user_test_data.copy(), user_id=user_id_to_tune)
                        logger.info(f"Evaluation for {user_id_to_tune}:\n{user_eval}")
                    else:
                         logger.info(f"No test data found for user {user_id_to_tune} to evaluate fine-tuned model.")
            else:
                logger.info(f"User {user_id_to_tune} has no data in processed training set for fine-tuning.")

        # 9. Save the Model Bundle
        logger.info(f"\n--- Saving Model Bundle to {MODEL_SAVE_PATH} ---")
        sleep_model.save_model(file_path=MODEL_SAVE_PATH)

        logger.info("--- Sleep Model Training Pipeline Completed Successfully ---")

    except FileNotFoundError as e:
        logger.error(f"Data file error: {e}", exc_info=True)
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error(f"Error during training pipeline: {str(e)}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred in the training pipeline: {str(e)}", exc_info=True)


if __name__ == "__main__":
    run_training_pipeline()

# --- End of train_evaluate.py ---
