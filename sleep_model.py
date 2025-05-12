# --- Imports ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
import lime
import lime.lime_tabular
import joblib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta

# --- Configure Logging (Only if running this file directly for tests) ---
# It's better to configure logging once in the main application entry point (app.py or train_evaluate.py)
# logger = logging.getLogger(__name__) # Get logger instance

# --- SleepQualityModel Class ---
class SleepQualityModel:
    """
    Manages the lifecycle of sleep quality prediction models, including
    data processing, training (global and user-specific), prediction,
    interpretation, and persistence.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False # Track if scaler is fitted
        self.mixed_effects_model_result = None

        self.global_alternative_models = {} # Stores RF, XGB trained on all data
        self.user_specific_models = {} # Stores models fine-tuned for user

        # Data for initializing explainers
        self.global_explainer_data_sample = None # Sample of X_train for global models
        self.user_specific_explainer_data = {} # Samples of X_train for user models

        # Define features, including placeholders for wearable data
        # Consider moving this to a config file or passing during init
        self.features_to_scale = [
            "ScreenTime", "StressLevel", "DietScore", "CircadianStability",
            "PhysicalActivity", "CaffeineIntake", "BedroomNoise", "BedroomLight",
            "EveningAlcohol", "ExerciseFrequency", "SocialJetlag", "MindfulnessPractice",
            "AvgHRV", "DeepSleepProportion", "StepsToday", "RestingHR"
        ]
        self.target = "SleepQuality"
        self.participant_id_col = "ParticipantID"
        self.date_col = "Date"

        # Get logger instance (assuming logging is configured elsewhere)
        self.logger = logging.getLogger(self.__class__.__name__)


    def load_data(self, file_path):
        """Loads data from a CSV file."""
        self.logger.info(f"Attempting to load data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
            self.logger.debug(f"Columns found: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def fit_scaler(self, df_train):
        """Fits the StandardScaler on the training data."""
        self.logger.info("Fitting StandardScaler...")
        features_present = [f for f in self.features_to_scale if f in df_train.columns]
        if not features_present:
            self.logger.error("No features found to fit the scaler.")
            raise ValueError("No features specified in 'features_to_scale' found in the training data.")

        try:
            self.scaler.fit(df_train[features_present])
            self.is_scaler_fitted = True
            self.logger.info(f"StandardScaler fitted on features: {features_present}")
        except Exception as e:
            self.logger.error(f"Error fitting scaler: {e}")
            self.is_scaler_fitted = False
            raise

    def preprocess_data(self, df, is_training_data=False):
        """
        Preprocesses the data: handles missing values, creates temporal features,
        and scales numerical features using the pre-fitted scaler.
        """
        self.logger.info(f"Starting data preprocessing... Is training data: {is_training_data}")
        df_processed = df.copy()

        # Ensure all expected features are present, add as NaN if missing
        for feature in self.features_to_scale:
            if feature not in df_processed.columns:
                self.logger.warning(
                    f"Feature '{feature}' not found in dataset. Adding as NaN."
                )
                df_processed[feature] = np.nan

        # Check for essential columns (target only needed for training/evaluation)
        if is_training_data and self.target not in df_processed.columns:
             raise ValueError(f"Target column '{self.target}' is missing in training data!")
        if self.participant_id_col not in df_processed.columns:
            # Allow prediction without participant ID, but log warning if needed for user models
            self.logger.debug(f"Participant ID column '{self.participant_id_col}' is missing.")
            # Add a dummy ID if needed downstream, or handle absence gracefully
            if self.participant_id_col not in df_processed.columns:
                 df_processed[self.participant_id_col] = "Unknown"


        # Impute missing values
        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            self.logger.warning(f"Found {missing_values.sum()} missing values. Imputing...")
            # Impute numerical with median, categorical with mode
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        # Use 0 for prediction if median isn't sensible without training data context
                        fill_value = df_processed[col].median() if is_training_data else 0
                        df_processed[col].fillna(fill_value, inplace=True)
                        self.logger.debug(f"Imputed NaNs in numeric column '{col}' with {fill_value}")
                    else: # Assuming categorical/object
                        mode_val = df_processed[col].mode()
                        fill_value = mode_val[0] if not mode_val.empty else "Unknown"
                        df_processed[col].fillna(fill_value, inplace=True)
                        self.logger.debug(f"Imputed NaNs in categorical column '{col}' with '{fill_value}'")

        # Temporal features
        if self.date_col in df_processed.columns:
            try:
                df_processed[self.date_col] = pd.to_datetime(df_processed[self.date_col], errors='coerce')
                # Check if conversion worked and column is not all NaT
                if pd.api.types.is_datetime64_any_dtype(df_processed[self.date_col]) and not df_processed[self.date_col].isnull().all():
                    df_processed["DayOfWeek"] = df_processed[self.date_col].dt.dayofweek
                    # Calculate DaysSinceStart relative to a fixed date or min date if available
                    min_date = df_processed[self.date_col].min() if not df_processed[self.date_col].isnull().all() else pd.Timestamp('now')
                    df_processed["DaysSinceStart"] = (df_processed[self.date_col] - min_date).dt.days
                    self.logger.info("Temporal features (DayOfWeek, DaysSinceStart) created.")
                else:
                    self.logger.warning(f"Could not create temporal features. '{self.date_col}' might be invalid or all NaT.")
                    df_processed["DayOfWeek"] = 0 # Default value
                    df_processed["DaysSinceStart"] = 0 # Default value
            except Exception as e:
                self.logger.error(f"Error creating temporal features: {str(e)}")
                df_processed["DayOfWeek"] = 0
                df_processed["DaysSinceStart"] = 0

        # Convert ParticipantID to category (useful for statsmodels)
        if self.participant_id_col in df_processed.columns:
            df_processed[self.participant_id_col] = df_processed[self.participant_id_col].astype("category")

        # Scale features using the FITTED scaler
        features_present_for_scaling = [
            f for f in self.features_to_scale if f in df_processed.columns
        ]
        if features_present_for_scaling:
            if self.is_scaler_fitted:
                try:
                    df_processed[features_present_for_scaling] = self.scaler.transform(
                        df_processed[features_present_for_scaling]
                    )
                    self.logger.info(f"Data transformed using pre-fitted scaler for features: {features_present_for_scaling}")
                except Exception as e:
                    self.logger.error(f"Error transforming data with scaler: {e}. Ensure scaler was fitted correctly and input columns match.")
                    raise
            else:
                # This should not happen if preprocess is called after fit_scaler for training
                # For prediction, it MUST be fitted.
                self.logger.error("Scaler is not fitted. Cannot transform data. Call fit_scaler() first.")
                raise RuntimeError("Scaler is not fitted. Cannot preprocess data requiring scaling.")
        else:
            self.logger.warning("No features to scale were found or specified in the input data.")

        self.logger.info(f"Preprocessing completed. Processed data shape: {df_processed.shape}")
        return df_processed

    # --- EDA and Visualization Methods (largely unchanged, ensure logger is used) ---
    def exploratory_data_analysis(self, df):
        self.logger.info("Performing exploratory data analysis...")
        # ... (rest of EDA logic using self.logger) ...
        if self.target not in df.columns:
            self.logger.error(f"Target column '{self.target}' not found for EDA.")
            return {}

        numeric_df_for_corr = df.select_dtypes(include=np.number)
        features_for_corr = [f for f in self.features_to_scale if f in numeric_df_for_corr.columns]

        eda_results = {
            'summary_statistics': df.describe(include='all'),
            'correlations': None, 'temporal_trends': None,
            'participant_variability': None, 'outlier_counts': {}
        }

        if self.target in numeric_df_for_corr.columns and features_for_corr:
            try:
                eda_results['correlations'] = numeric_df_for_corr[features_for_corr + [self.target]].corr(numeric_only=True)[self.target].sort_values(ascending=False)
            except Exception as e:
                self.logger.warning(f"Could not compute correlations: {e}")
        self.logger.info("Exploratory data analysis completed.")
        return eda_results


    def visualize_data(self, df):
        self.logger.info("Creating visualizations...")
        # ... (rest of visualization logic using self.logger) ...
        if self.target not in df.columns:
            self.logger.error(f"Target column '{self.target}' not found for visualization.")
            return

        try:
            plt.figure(figsize=(10, 6)) # Adjusted size
            sns.histplot(df[self.target], kde=True, bins=10) # Fewer bins might be clearer
            plt.title(f'{self.target} Distribution')
            plt.xlabel(self.target)
            plt.ylabel('Frequency')
            plt.tight_layout()
            # In a script, save the figure instead of showing
            # plt.savefig("sleep_quality_distribution.png")
            # plt.close() # Close the plot to free memory
            self.logger.info(f"Histogram for '{self.target}' created.")
            # Add more plots as needed (scatter plots, box plots per participant etc.)

        except Exception as e:
            self.logger.error(f"Error during visualization: {e}")


    # --- Data Splitting Method (unchanged, ensure logger is used) ---
    def split_data(self, df, test_size=0.2, temporal=False, test_days=7):
        self.logger.info(f"Splitting data. Test size: {test_size}, Temporal: {temporal}, Test days: {test_days}")
        # ... (rest of splitting logic using self.logger) ...
        if self.participant_id_col not in df.columns:
            self.logger.error("Participant ID column is required for splitting.")
            raise ValueError("Participant ID column is required for splitting.")

        # Temporal Split Logic
        if temporal:
            if self.date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                self.logger.warning(f"Date column '{self.date_col}' not found or not datetime. Using random split.")
                temporal = False
            else:
                df_temp_split = df.dropna(subset=[self.date_col]).sort_values(by=self.date_col)
                if df_temp_split.empty or len(df_temp_split[self.date_col].unique()) < test_days + 1:
                    self.logger.warning("Not enough data or unique dates for temporal split. Falling back to random.")
                    temporal = False
                else:
                    cutoff_date = df_temp_split[self.date_col].max() - timedelta(days=test_days)
                    train_df = df_temp_split[df_temp_split[self.date_col] <= cutoff_date]
                    test_df = df_temp_split[df_temp_split[self.date_col] > cutoff_date]
                    if train_df.empty or test_df.empty:
                        self.logger.warning("Temporal split resulted in empty train/test. Falling back to random.")
                        temporal = False
                    else:
                        self.logger.info(f"Temporal split successful: {len(train_df)} train, {len(test_df)} test.")
                        return train_df, test_df

        # Participant-Based Random Split Logic (Fallback or if temporal=False)
        unique_participants = df[self.participant_id_col].unique()
        if len(unique_participants) < 2:
            self.logger.warning("Not enough unique participants (< 2) for robust split. Using random split on all data or returning as is if too small.")
            if len(df) < 10: # Avoid splitting tiny datasets
                 self.logger.warning("Dataset too small (< 10 rows), returning full dataset as train, empty as test.")
                 return df, pd.DataFrame(columns=df.columns)
            # Fallback to simple random split if only one participant or very few
            return train_test_split(df, test_size=test_size, random_state=self.random_state)

        # Proceed with participant-based split
        train_participants, test_participants = train_test_split(
            unique_participants, test_size=test_size, random_state=self.random_state
        )
        train_df = df[df[self.participant_id_col].isin(train_participants)]
        test_df = df[df[self.participant_id_col].isin(test_participants)]
        self.logger.info(f"Participant-based random split: {len(train_df)} train ({len(train_participants)} users), {len(test_df)} test ({len(test_participants)} users).")
        return train_df, test_df


    # --- Model Training Methods (largely unchanged, ensure logger is used) ---
    def fit_global_models(self, train_df):
        """Fits global models (Mixed Effects, RF, XGB) on the training data."""
        self.logger.info("Fitting global predictive models...")
        # Data should already be preprocessed (scaled) before passing here
        X_train = train_df[self.features_to_scale]
        y_train = train_df[self.target]

        if X_train.empty or y_train.empty:
            self.logger.error("Training data (X_train or y_train) is empty. Cannot fit global models.")
            return

        # Store sample for global explainers (use original scaled data)
        sample_size = min(100, len(X_train))
        self.global_explainer_data_sample = X_train.sample(
            n=sample_size, random_state=self.random_state
        )
        self.logger.info(f"Stored sample ({sample_size} rows) of training data for global explainers.")

        # --- Mixed Effects Model ---
        try:
            # Formula requires original (unscaled) or specially handled data if using scaled
            # Using scaled data here for consistency, but coefficients interpretation changes
            formula = f"{self.target} ~ {' + '.join(self.features_to_scale)}"
            # Ensure ParticipantID is present for grouping
            if self.participant_id_col not in train_df.columns:
                 raise ValueError(f"'{self.participant_id_col}' column needed for mixed effects model groups.")

            me_model = mixedlm(formula, train_df, groups=train_df[self.participant_id_col])
            self.mixed_effects_model_result = me_model.fit()
            self.logger.info("Global Mixed effects model fitted successfully.")
            self.logger.debug(f"ME Model Summary:\n{self.mixed_effects_model_result.summary()}")
        except Exception as e:
            self.logger.error(f"Error fitting global mixed effects model: {str(e)}")
            self.mixed_effects_model_result = None # Ensure it's None on failure

        # --- Random Forest Regressor ---
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1 # Use all cores
            )
            rf_model.fit(X_train, y_train)
            self.global_alternative_models["random_forest"] = rf_model
            self.logger.info("Global Random Forest model fitted successfully.")
        except Exception as e:
            self.logger.error(f"Error fitting global Random Forest model: {str(e)}")

        # --- XGBoost Regressor ---
        try:
            xgb_model = XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, objective="reg:squarederror",
                n_jobs=-1 # Use all cores
            )
            xgb_model.fit(X_train, y_train)
            self.global_alternative_models["xgboost"] = xgb_model
            self.logger.info("Global XGBoost model fitted successfully.")
        except Exception as e:
            self.logger.error(f"Error fitting global XGBoost model: {str(e)}")


    def fine_tune_model_for_user(self, user_id, user_df, model_type="random_forest", min_samples=15):
        """Fine-tunes a model (RF or XGB) for a specific user."""
        self.logger.info(f"Attempting to fine-tune {model_type} for user {user_id}")
        # user_df should already be preprocessed (scaled)
        if len(user_df) < min_samples:
            self.logger.warning(
                f"Not enough data ({len(user_df)} points, need {min_samples}) for user {user_id} to fine-tune {model_type}. Skipping."
            )
            return False

        X_user = user_df[self.features_to_scale]
        y_user = user_df[self.target]

        if X_user.empty or y_user.empty:
            self.logger.error(f"User data (X or y) is empty for user {user_id}. Cannot fine-tune.")
            return False

        user_model_key = f"{user_id}_{model_type}"
        user_model_instance = None

        try:
            # Define model parameters (could be different from global)
            if model_type == "random_forest":
                user_model_instance = RandomForestRegressor(
                    n_estimators=50, max_depth=7, random_state=self.random_state, n_jobs=-1
                )
            elif model_type == "xgboost":
                user_model_instance = XGBRegressor(
                    n_estimators=50, max_depth=5, learning_rate=0.1,
                    random_state=self.random_state, objective="reg:squarederror", n_jobs=-1
                )
            else:
                self.logger.error(f"Unsupported model_type '{model_type}' for fine-tuning.")
                return False

            user_model_instance.fit(X_user, y_user)
            self.user_specific_models[user_model_key] = user_model_instance

            # Store sample data for this user-specific model's explainer
            sample_size = min(max(1, len(X_user) // 2), 50) # Ensure at least 1, up to 50
            self.user_specific_explainer_data[user_model_key] = X_user.sample(
                n=sample_size, random_state=self.random_state
            )
            self.logger.info(f"Successfully fine-tuned and stored model and explainer data for {user_model_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error fine-tuning {model_type} for user {user_id}: {e}")
            # Clean up if failed
            if user_model_key in self.user_specific_models:
                del self.user_specific_models[user_model_key]
            if user_model_key in self.user_specific_explainer_data:
                del self.user_specific_explainer_data[user_model_key]
            return False

    # --- Evaluation Methods (largely unchanged, ensure logger is used) ---
    def evaluate_models(self, test_df, user_id=None):
        """Evaluates global or user-specific models on the test data."""
        # test_df should be preprocessed (scaled)
        evaluation_results = {}
        required_cols = self.features_to_scale + [self.target]
        if not all(f in test_df.columns for f in required_cols):
            self.logger.error(f"Test data missing required columns ({required_cols}). Cannot evaluate.")
            return {}

        X_test = test_df[self.features_to_scale]
        y_test = test_df[self.target]

        if X_test.empty or y_test.empty:
            self.logger.warning("Test data (X_test or y_test) is empty. Cannot evaluate.")
            return {}

        models_to_evaluate = {}
        eval_context = "global"
        if user_id:
            eval_context = f"user-specific ({user_id})"
            for model_key, model_instance in self.user_specific_models.items():
                if model_key.startswith(f"{user_id}_"):
                    models_to_evaluate[model_key] = model_instance
            if not models_to_evaluate:
                self.logger.warning(f"No specific models found for user {user_id} to evaluate.")
                # Optionally evaluate global models as fallback?
                # return self.evaluate_models(test_df, user_id=None) # Be careful of recursion
                return {} # Return empty if no user models found
        else: # Evaluate global models
            models_to_evaluate = self.global_alternative_models.copy() # Evaluate copies
            # Add Mixed Effects model evaluation if available
            if self.mixed_effects_model_result:
                try:
                    # Predict using fixed effects (population average)
                    # Note: This doesn't use random effects for prediction on new data easily
                    params = self.mixed_effects_model_result.params
                    intercept = params.get('Intercept', 0)
                    predictions_me = pd.Series(intercept, index=X_test.index)
                    for feature in self.features_to_scale:
                        if feature in params:
                             # Ensure feature exists in X_test (should if required_cols check passed)
                             if feature in X_test.columns:
                                 predictions_me += params[feature] * X_test[feature]
                             else:
                                 self.logger.warning(f"Feature '{feature}' from ME model params not in X_test during evaluation.")

                    evaluation_results['mixed_effects (global fixed)'] = {
                        'mse': mean_squared_error(y_test, predictions_me),
                        'rmse': np.sqrt(mean_squared_error(y_test, predictions_me)),
                        'r2': r2_score(y_test, predictions_me),
                        'mae': mean_absolute_error(y_test, predictions_me)
                    }
                except Exception as e:
                    self.logger.error(f"Error evaluating mixed effects model: {str(e)}")
                    evaluation_results['mixed_effects (global fixed)'] = {'error': str(e)}

        self.logger.info(f"Evaluating {eval_context} models...")
        for name, model_instance in models_to_evaluate.items():
            try:
                preds = model_instance.predict(X_test)
                evaluation_results[name] = {
                    "mse": mean_squared_error(y_test, preds),
                    "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                    "r2": r2_score(y_test, preds),
                    "mae": mean_absolute_error(y_test, preds),
                }
            except Exception as e:
                self.logger.error(f"Error evaluating {name} model: {str(e)}")
                evaluation_results[name] = {"error": str(e)}

        # Log summary
        self.logger.info(f"--- Evaluation Summary ({eval_context}) ---")
        for model_name, metrics in evaluation_results.items():
            if "error" not in metrics:
                self.logger.info(
                    f"  {model_name}: RMSE={metrics.get('rmse', float('nan')):.3f}, R²={metrics.get('r2', float('nan')):.3f}, MAE={metrics.get('mae', float('nan')):.3f}"
                )
            else:
                 self.logger.warning(f"  {model_name}: Evaluation Error - {metrics['error']}")
        self.logger.info("--- End Evaluation Summary ---")
        return evaluation_results

    def cross_validate_global_models(self, df, n_splits=5):
        """Performs K-Fold cross-validation for global RF and XGB models."""
        self.logger.info(f"Starting {n_splits}-fold cross-validation for global models...")
        # df should be preprocessed (scaled)
        required_cols = self.features_to_scale + [self.target]
        if not all(f in df.columns for f in required_cols):
            self.logger.error(f"Data for CV missing required columns ({required_cols}). Cannot perform CV.")
            return {}

        X = df[self.features_to_scale]
        y = df[self.target]
        # Use KFold for standard CV. For time series or grouped data, consider TimeSeriesSplit or GroupKFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        cv_results = {name: {'rmse': [], 'r2': [], 'mae': []} for name in self.global_alternative_models.keys()}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            self.logger.info(f"--- CV Fold {fold+1}/{n_splits} ---")
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Cross-validate Random Forest
            if 'random_forest' in self.global_alternative_models:
                try:
                    # Re-initialize model for each fold to avoid data leakage across folds
                    rf_fold = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)
                    rf_fold.fit(X_train_fold, y_train_fold)
                    preds_rf = rf_fold.predict(X_val_fold)
                    cv_results['random_forest']['rmse'].append(np.sqrt(mean_squared_error(y_val_fold, preds_rf)))
                    cv_results['random_forest']['r2'].append(r2_score(y_val_fold, preds_rf))
                    cv_results['random_forest']['mae'].append(mean_absolute_error(y_val_fold, preds_rf))
                    self.logger.debug(f"  RF Fold {fold+1} completed.")
                except Exception as e:
                    self.logger.error(f"  Error in RF CV fold {fold+1}: {e}")

            # Cross-validate XGBoost
            if 'xgboost' in self.global_alternative_models:
                try:
                    xgb_fold = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                       random_state=self.random_state, objective='reg:squarederror', n_jobs=-1)
                    xgb_fold.fit(X_train_fold, y_train_fold)
                    preds_xgb = xgb_fold.predict(X_val_fold)
                    cv_results['xgboost']['rmse'].append(np.sqrt(mean_squared_error(y_val_fold, preds_xgb)))
                    cv_results['xgboost']['r2'].append(r2_score(y_val_fold, preds_xgb))
                    cv_results['xgboost']['mae'].append(mean_absolute_error(y_val_fold, preds_xgb))
                    self.logger.debug(f"  XGB Fold {fold+1} completed.")
                except Exception as e:
                    self.logger.error(f"  Error in XGB CV fold {fold+1}: {e}")

        # Summarize CV results
        self.logger.info("--- Cross-Validation Summary ---")
        summary = {}
        for model_name_cv, results in cv_results.items():
            if results['rmse']: # Check if list is not empty (i.e., folds ran successfully)
                summary[model_name_cv] = {
                    'mean_rmse': np.mean(results['rmse']),
                    'std_rmse': np.std(results['rmse']),
                    'mean_r2': np.mean(results['r2']),
                    'std_r2': np.std(results['r2']),
                    'mean_mae': np.mean(results['mae']),
                    'std_mae': np.std(results['mae']),
                }
                self.logger.info(f"  {model_name_cv}:")
                self.logger.info(f"    RMSE: {summary[model_name_cv]['mean_rmse']:.3f} ± {summary[model_name_cv]['std_rmse']:.3f}")
                self.logger.info(f"    R²:   {summary[model_name_cv]['mean_r2']:.3f} ± {summary[model_name_cv]['std_r2']:.3f}")
                self.logger.info(f"    MAE:  {summary[model_name_cv]['mean_mae']:.3f} ± {summary[model_name_cv]['std_mae']:.3f}")
            else:
                summary[model_name_cv] = {'error': 'No successful folds or model not included in CV'}
                self.logger.warning(f"  {model_name_cv}: No successful CV folds.")
        self.logger.info("--- End Cross-Validation Summary ---")
        return summary


    # --- Prediction Method ---
    def predict_sleep_quality(self, new_data_input, model_type="random_forest", user_id=None):
        """
        Predicts sleep quality for new input data. Handles dict or DataFrame input.
        Input data should be raw (unscaled).
        """
        self.logger.info(f"Received prediction request. Model type: {model_type}, User ID: {user_id}")
        if isinstance(new_data_input, dict):
            new_data_df = pd.DataFrame([new_data_input])
        elif isinstance(new_data_input, pd.DataFrame):
            new_data_df = new_data_input.copy()
        else:
            self.logger.error(f"Invalid input type for prediction: {type(new_data_input)}. Must be dict or DataFrame.")
            raise TypeError("Input data must be a dict or DataFrame.")

        # Preprocess the input data (handles missing cols, imputation, scaling)
        # Pass is_training_data=False
        try:
            processed_df = self.preprocess_data(new_data_df, is_training_data=False)
        except Exception as e:
             self.logger.error(f"Error preprocessing input data for prediction: {e}")
             raise # Re-raise the exception

        # Select only the features needed for prediction
        features_for_prediction = processed_df[self.features_to_scale]

        # --- Model Selection Logic ---
        model_to_use = None
        model_key_used = ""
        is_user_specific = False

        if user_id:
            user_model_key = f"{user_id}_{model_type}"
            if user_model_key in self.user_specific_models:
                model_to_use = self.user_specific_models[user_model_key]
                model_key_used = f"user-specific: {user_model_key}"
                is_user_specific = True
                self.logger.info(f"Using user-specific model: {user_model_key}")
            else:
                self.logger.warning(
                    f"User-specific model {user_model_key} not found. Falling back to global {model_type}."
                )

        # Fallback to global model if no user_id or user-specific model not found/applicable
        if model_to_use is None:
            if model_type in self.global_alternative_models:
                model_to_use = self.global_alternative_models[model_type]
                model_key_used = f"global: {model_type}"
                self.logger.info(f"Using global model: {model_type}")
            elif model_type == "mixed_effects" and self.mixed_effects_model_result:
                # Handle mixed effects prediction (using fixed effects)
                self.logger.info("Using global: mixed_effects model (fixed effects prediction)")
                try:
                    params = self.mixed_effects_model_result.params
                    intercept = params.get('Intercept', 0)
                    # Ensure features_for_prediction has the right columns
                    predictions = pd.Series(intercept, index=features_for_prediction.index)
                    for feature in self.features_to_scale:
                        if feature in params and feature in features_for_prediction.columns:
                            predictions += params[feature] * features_for_prediction[feature]
                    model_key_used = "global: mixed_effects (fixed effects)"
                    # Return result based on input size
                    result = predictions.tolist()
                    self.logger.info(f"Prediction successful using {model_key_used}.")
                    return result[0] if len(result) == 1 else result
                except Exception as e:
                    self.logger.error(f"Error predicting with mixed effects model: {e}")
                    raise ValueError(f"Failed to predict using mixed_effects model: {e}")
            else:
                self.logger.error(f"Model type '{model_type}' (global or user-specific for {user_id}) is not available or trained.")
                raise ValueError(f"Model '{model_type}' not available.")

        # Check if a model was successfully selected
        if model_to_use is None:
             self.logger.critical("Logical error: No model selected for prediction despite checks.")
             raise ValueError(f"Could not find a suitable model for prediction (type: {model_type}, user: {user_id}).")

        # --- Perform Prediction ---
        try:
            self.logger.debug(f"Predicting with model: {model_key_used}")
            predictions = model_to_use.predict(features_for_prediction)
            # Ensure output is standard Python type (float or list of floats)
            result = predictions.tolist()
            self.logger.info(f"Prediction successful using {model_key_used}.")
            # Return single float if single prediction, else list
            return result[0] if len(result) == 1 else result
        except Exception as e:
            self.logger.error(f"Error during prediction with model {model_key_used}: {e}")
            raise # Re-raise the exception


    # --- Interpretation Method ---
    def interpret_model(self, instance_input, model_type="random_forest", user_id=None):
        """
        Generates SHAP and LIME explanations for a single instance.
        Input instance should be raw (unscaled).
        """
        self.logger.info(f"Received interpretation request. Model type: {model_type}, User ID: {user_id}")
        # --- Prepare Instance ---
        if isinstance(instance_input, dict):
            instance_df = pd.DataFrame([instance_input])
        elif isinstance(instance_input, pd.DataFrame):
            instance_df = instance_input.iloc[[0]].copy() # Ensure single instance
        else:
            self.logger.error(f"Invalid input type for interpretation: {type(instance_input)}. Must be dict or single-row DataFrame.")
            raise TypeError("Input instance must be a dict or single-row DataFrame.")

        # Preprocess the instance (handles missing cols, imputation, scaling)
        try:
            processed_instance_df = self.preprocess_data(instance_df, is_training_data=False)
        except Exception as e:
             self.logger.error(f"Error preprocessing instance for interpretation: {e}")
             return {"error": f"Preprocessing failed: {e}"}

        # Extract scaled features as DataFrame and NumPy array
        instance_scaled_df = processed_instance_df[self.features_to_scale]
        if instance_scaled_df.empty:
             self.logger.error("Scaled instance data is empty after preprocessing.")
             return {"error": "Scaled instance data is empty."}
        # LIME needs a 1D numpy array for the instance
        instance_scaled_numpy = instance_scaled_df.iloc[0].values


        # --- Select Model and Background Data ---
        model_to_explain = None
        background_data_for_explainer = None # Should be DataFrame
        model_key_used = ""
        is_user_specific = False

        if user_id:
            user_model_key = f"{user_id}_{model_type}"
            if user_model_key in self.user_specific_models:
                model_to_explain = self.user_specific_models[user_model_key]
                if user_model_key in self.user_specific_explainer_data:
                    background_data_for_explainer = self.user_specific_explainer_data[user_model_key]
                model_key_used = f"user-specific: {user_model_key}"
                is_user_specific = True
                self.logger.info(f"Using user-specific model for interpretation: {user_model_key}")
            else:
                self.logger.warning(f"User-specific model {user_model_key} not found for interpretation. Falling back to global.")

        if model_to_explain is None: # Fallback or no user_id
            if model_type in self.global_alternative_models:
                model_to_explain = self.global_alternative_models[model_type]
                background_data_for_explainer = self.global_explainer_data_sample
                model_key_used = f"global: {model_type}"
                self.logger.info(f"Using global model for interpretation: {model_type}")
            # Add mixed_effects interpretation if desired (less common with SHAP/LIME)
            # elif model_type == "mixed_effects": ...

        # --- Validate Model and Background Data ---
        if model_to_explain is None:
            msg = f"Model {model_type} (global or user-specific for {user_id}) not found for interpretation."
            self.logger.error(msg)
            return {"error": msg}
        if background_data_for_explainer is None or background_data_for_explainer.empty:
            msg = f"Background data for explainer of {model_key_used} is missing or empty."
            self.logger.error(msg)
            # Fallback: Use the instance itself? Less ideal.
            # background_data_for_explainer = instance_scaled_df
            return {"error": msg}
        if not isinstance(background_data_for_explainer, pd.DataFrame):
             msg = f"Background data for {model_key_used} is not a DataFrame (type: {type(background_data_for_explainer)})."
             self.logger.error(msg)
             return {"error": msg}


        # --- Generate Explanations ---
        self.logger.info(f"Interpreting model {model_key_used} using SHAP and LIME.")
        explanations = {'model_interpreted': model_key_used}
        feature_names = background_data_for_explainer.columns.tolist() # Use names from background data

        # SHAP Explanations
        try:
            self.logger.debug("Initializing SHAP Explainer...")
            # Use KernelExplainer for broader compatibility, though TreeExplainer is faster for RF/XGB
            # shap_explainer = shap.KernelExplainer(model_to_explain.predict, background_data_for_explainer)
            # Or use Explainer which auto-detects
            shap_explainer = shap.Explainer(model_to_explain, background_data_for_explainer)
            self.logger.debug("Calculating SHAP values...")
            shap_values = shap_explainer(instance_scaled_df) # Pass DataFrame

            # Extract base value and shap values for the instance
            # SHAP output structure can vary slightly based on explainer type and model
            if hasattr(shap_values, 'values') and hasattr(shap_values, 'base_values'):
                 # Common structure for TreeExplainer, KernelExplainer etc. on single output models
                 instance_shap_values = shap_values.values[0]
                 instance_base_value = shap_values.base_values[0]
            else:
                 # Fallback or different structure? Inspect shap_values object
                 self.logger.warning(f"Unexpected SHAP output structure: {type(shap_values)}. Attempting basic extraction.")
                 # This might need adjustment based on the specific shap_values object received
                 instance_shap_values = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values
                 instance_base_value = None # Base value might not be directly available

            explanations["shap_values"] = instance_shap_values.tolist()
            explanations["shap_base_value"] = float(instance_base_value) if instance_base_value is not None else None
            explanations["shap_feature_names"] = feature_names # Use consistent feature names
            self.logger.info("SHAP explanation generated successfully.")

        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation for {model_key_used}: {e}", exc_info=True)
            explanations["shap_error"] = str(e)

        # LIME Explanations
        try:
            self.logger.debug("Initializing LIME Explainer...")
            # Wrapper for the predict_fn for LIME, ensuring DataFrame input
            def lime_predict_fn_wrapper(numpy_data):
                # Convert numpy array back to DataFrame with correct feature names
                if numpy_data.ndim == 1:
                    numpy_data_2d = numpy_data.reshape(1, -1)
                else:
                    numpy_data_2d = numpy_data

                # Ensure columns match the background data used for the explainer
                df_for_predict = pd.DataFrame(numpy_data_2d, columns=feature_names)
                try:
                    predictions = model_to_explain.predict(df_for_predict)
                    return predictions
                except Exception as lime_pred_e:
                    self.logger.error(f"Error inside LIME predict_fn wrapper: {lime_pred_e}")
                    # Return array of NaNs or zeros of expected shape if prediction fails
                    return np.full(len(numpy_data_2d), np.nan)


            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=background_data_for_explainer.values, # LIME expects numpy array
                feature_names=feature_names,
                mode="regression",
                random_state=self.random_state,
                # class_names=[self.target] # Optional: name the output
            )
            self.logger.debug("Explaining instance with LIME...")
            lime_exp = lime_explainer.explain_instance(
                data_row=instance_scaled_numpy, # Pass the 1D numpy array instance
                predict_fn=lime_predict_fn_wrapper,
                num_features=len(feature_names), # Explain all features
            )
            # Store as list of tuples (feature, weight) and the intercept
            explanations["lime_explanation"] = lime_exp.as_list()
            explanations["lime_intercept"] = lime_exp.intercept[0] # Regression intercept
            explanations["lime_r2_score"] = lime_exp.score # R2 score of the local LIME model
            self.logger.info(f"LIME explanation generated successfully (R2={lime_exp.score:.3f}).")

        except Exception as e:
            self.logger.error(f"Error generating LIME explanation for {model_key_used}: {e}", exc_info=True)
            explanations["lime_error"] = str(e)

        return explanations


    # --- Save/Load Methods ---
    def save_model(self, file_path="sleep_model_bundle.pkl"):
        """Saves the entire model state (models, scaler, data) to a file."""
        self.logger.info(f"Saving model bundle to {file_path}...")
        model_data = {
            "random_state": self.random_state,
            "scaler": self.scaler,
            "is_scaler_fitted": self.is_scaler_fitted, # Save scaler status
            "mixed_effects_model_result": self.mixed_effects_model_result,
            "global_alternative_models": self.global_alternative_models,
            "user_specific_models": self.user_specific_models,
            "global_explainer_data_sample": self.global_explainer_data_sample,
            "user_specific_explainer_data": self.user_specific_explainer_data,
            "features_to_scale": self.features_to_scale,
            "target": self.target,
            "participant_id_col": self.participant_id_col,
            "date_col": self.date_col,
        }
        try:
            # Consider compression level for large models/data
            joblib.dump(model_data, file_path, compress=3)
            self.logger.info(f"Model bundle saved successfully to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model bundle: {str(e)}")
            raise

    @staticmethod
    def load_model(file_path="sleep_model_bundle.pkl"):
        """Loads the model state from a file."""
        # Get a logger instance for the static method
        static_logger = logging.getLogger(f"{SleepQualityModel.__name__}.load_model")
        static_logger.info(f"Loading model bundle from {file_path}...")
        try:
            model_data = joblib.load(file_path)
            # Create a new instance and populate it
            model = SleepQualityModel(random_state=model_data.get("random_state", 42))

            # Load components, checking for existence
            model.scaler = model_data.get("scaler")
            model.is_scaler_fitted = model_data.get("is_scaler_fitted", False)
            model.mixed_effects_model_result = model_data.get("mixed_effects_model_result")
            model.global_alternative_models = model_data.get("global_alternative_models", {})
            model.user_specific_models = model_data.get("user_specific_models", {})
            model.global_explainer_data_sample = model_data.get("global_explainer_data_sample")
            model.user_specific_explainer_data = model_data.get("user_specific_explainer_data", {})
            model.features_to_scale = model_data.get("features_to_scale", [])
            model.target = model_data.get("target")
            model.participant_id_col = model_data.get("participant_id_col")
            model.date_col = model_data.get("date_col")

            # --- Post-load checks ---
            if not model.features_to_scale:
                 static_logger.warning("Loaded model has no 'features_to_scale'. This might cause issues.")
            if model.scaler is None:
                 static_logger.warning("Loaded model has no 'scaler' object.")
            elif not model.is_scaler_fitted:
                 static_logger.warning("Loaded model's scaler is present but marked as 'not fitted'. Predictions requiring scaling might fail.")
            if not model.global_alternative_models and not model.user_specific_models and not model.mixed_effects_model_result:
                 static_logger.warning("Loaded model has no trained predictive models (global, user-specific, or mixed-effects).")

            static_logger.info(f"Model bundle loaded successfully from {file_path}")
            return model
        except FileNotFoundError:
             static_logger.error(f"Model file not found: {file_path}")
             raise
        except Exception as e:
            static_logger.error(f"Error loading model bundle from {file_path}: {str(e)}")
            raise

# --- End of sleep_model.py ---
