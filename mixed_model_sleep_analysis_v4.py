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
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SleepQualityModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.mixed_effects_model_result = None # For statsmodels mixedlm

        self.global_alternative_models = {} # Stores RF, XGB trained on all data
        self.user_specific_models = {} # Stores models fine-tuned for users
        # { 'user1_rf': model, 'user2_xgb': model }

        # Data for initializing explainers
        self.global_explainer_data_sample = None # Sample of X_train for global models
        self.user_specific_explainer_data = {} # Samples of X_train for user models
        # { 'user1_rf': df_sample, 'user2_xgb': df_sample }

        # Define features, including placeholders for wearable data
        self.features_to_scale = [
            "ScreenTime", "StressLevel", "DietScore", "CircadianStability",
            "PhysicalActivity", "CaffeineIntake", "BedroomNoise", "BedroomLight",
            "EveningAlcohol", "ExerciseFrequency", "SocialJetlag", "MindfulnessPractice",
            # Potential Wearable Data (will be NaN if not present and imputed)
            "AvgHRV", "DeepSleepProportion", "StepsToday", "RestingHR"
        ]
        self.target = "SleepQuality"
        self.participant_id_col = "ParticipantID" # Renamed for clarity
        self.date_col = "Date"

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns found: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def preprocess_data(self, df):
        logger.info("Starting enhanced data preprocessing...")
        df_processed = df.copy()

        # Ensure all expected features are present, add as NaN if missing
        for feature in self.features_to_scale:
            if feature not in df_processed.columns:
                logger.warning(
                    f"Feature '{feature}' not found in dataset. Adding as NaN."
                )
                df_processed[feature] = np.nan

        if self.target not in df_processed.columns:
            raise ValueError(f"Target column '{self.target}' is missing!")
        if self.participant_id_col not in df_processed.columns:
            raise ValueError(
                f"Participant ID column '{self.participant_id_col}' is missing!"
            )

        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found {missing_values.sum()} missing values.")
            logger.info(
                "Imputing missing values: median for numerical, mode for categorical."
            )
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col].fillna(
                            df_processed[col].median(), inplace=True
                        )
                    else:
                        mode_val = df_processed[col].mode()
                        if not mode_val.empty:
                            df_processed[col].fillna(mode_val[0], inplace=True)
                        else:
                            df_processed[col].fillna("Unknown", inplace=True)

        if self.date_col in df_processed.columns:
            try:
                df_processed[self.date_col] = pd.to_datetime(
                    df_processed[self.date_col], errors="coerce"
                )
                if not df_processed[self.date_col].isnull().all():
                    df_processed["DayOfWeek"] = df_processed[
                        self.date_col
                    ].dt.dayofweek
                    df_processed["DaysSinceStart"] = (
                        df_processed[self.date_col]
                        - df_processed[self.date_col].min()
                    ).dt.days
                else:
                    df_processed["DayOfWeek"] = 0
                    df_processed["DaysSinceStart"] = 0
                logger.info("Temporal features created successfully.")
            except Exception as e:
                logger.error(f"Error creating temporal features: {str(e)}")

        df_processed[self.participant_id_col] = df_processed[
            self.participant_id_col
        ].astype("category")

        # Fit scaler on all specified features if present, transform later
        features_present_for_scaling = [
            f for f in self.features_to_scale if f in df_processed.columns
        ]
        if features_present_for_scaling:
            # Fit the scaler on the full dataset (or training part if split first)
            # Here, we fit on the whole df before splitting for simplicity in this example
            # In a strict train/test setup, fit ONLY on training data.
            self.scaler.fit(df_processed[features_present_for_scaling])
            df_processed[features_present_for_scaling] = self.scaler.transform(
                df_processed[features_present_for_scaling]
            )
            logger.info(
                f"Scaler fitted and data transformed for: {features_present_for_scaling}"
            )
        else:
            logger.warning("No features to scale were found or specified.")

        logger.info(f"Preprocessed data shape: {df_processed.shape}")
        logger.info("Enhanced data preprocessing completed.")
        return df_processed

    def exploratory_data_analysis(self, df): # df here is pre-scaled
        logger.info("Performing comprehensive exploratory data analysis...")
        # ... (implementation largely unchanged, ensure it uses scaled features if appropriate for EDA)
        # For correlations, it's often better to use data *before* scaling if you want to interpret magnitudes.
        # However, for consistency with model inputs, using scaled data is also an option.
        # I'll assume df is the output of preprocess_data (i.e., scaled)
        if self.target not in df.columns:
            logger.error(f"Target column '{self.target}' not found for EDA.")
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
                logger.warning(f"Could not compute correlations: {e}")
        # ... (rest of EDA remains similar)
        logger.info("Exploratory data analysis completed.")
        return eda_results


    def visualize_data(self, df): # df here is pre-scaled
        logger.info("Creating comprehensive visualizations...")
        # ... (implementation largely unchanged, ensure it uses scaled features if appropriate)
        if self.target not in df.columns:
            logger.error(f"Target column '{self.target}' not found for visualization.")
            return

        plt.figure(figsize=(18, 12))
        plot_index = 1

        plt.subplot(2, 3, plot_index); plot_index += 1
        sns.histplot(df[self.target], kde=True)
        plt.title('Sleep Quality Distribution')
        # ... (rest of visualization remains similar)
        plt.tight_layout()
        # plt.show() # Call this explicitly in main if needed
        logger.info("Visualizations created. Call plt.show() to display.")


    def split_data(self, df, test_size=0.2, temporal=False, test_days=7):
        # ... (implementation largely unchanged, uses self.participant_id_col)
        if self.participant_id_col not in df.columns:
            raise ValueError("Participant ID column is required for splitting.")

        if temporal:
            if self.date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                logger.warning(f"Date column '{self.date_col}' not found or not datetime. Using random split.")
                temporal = False
            else:
                df_temp_split = df.dropna(subset=[self.date_col])
                if df_temp_split.empty or len(df_temp_split[self.date_col].unique()) < test_days + 1:
                    logger.warning("Not enough data or unique dates for temporal split. Falling back to random.")
                    temporal = False
                else:
                    cutoff_date = df_temp_split[self.date_col].max() - timedelta(days=test_days)
                    train_df = df_temp_split[df_temp_split[self.date_col] <= cutoff_date]
                    test_df = df_temp_split[df_temp_split[self.date_col] > cutoff_date]
                    if train_df.empty or test_df.empty:
                        logger.warning("Temporal split resulted in empty train/test. Falling back to random.")
                        temporal = False
                    else:
                        logger.info(f"Temporal split: {len(train_df)} train, {len(test_df)} test.")
                        return train_df, test_df

        unique_participants = df[self.participant_id_col].unique()
        if len(unique_participants) < 2: # Need at least one for train and one for test for participant-based split
            logger.warning("Not enough unique participants for robust split. Using random split on all data or returning as is if too small.")
            if len(df) < 10: # Arbitrary small number
                 return df, pd.DataFrame(columns=df.columns) # Not enough to split
            # Fallback to simple random split if participant split is not feasible
            return train_test_split(df, test_size=test_size, random_state=self.random_state)


        train_participants, test_participants = train_test_split(
            unique_participants, test_size=test_size, random_state=self.random_state
        )
        train_df = df[df[self.participant_id_col].isin(train_participants)]
        test_df = df[df[self.participant_id_col].isin(test_participants)]
        logger.info(f"Participant-based random split: {len(train_df)} train, {len(test_df)} test.")
        return train_df, test_df


    def fit_global_models(self, train_df):
        logger.info("Fitting global predictive models...")
        # Ensure all features_to_scale are present in train_df (they should be, due to preprocessing)
        X_train = train_df[self.features_to_scale]
        y_train = train_df[self.target]

        if X_train.empty or y_train.empty:
            logger.error("Training data (X_train or y_train) is empty. Cannot fit global models.")
            return

        # Store a sample of the training data for global model explainers
        sample_size = min(100, len(X_train))
        self.global_explainer_data_sample = X_train.sample(
            n=sample_size, random_state=self.random_state
        )

        # --- Mixed Effects Model (Optional) ---
        try:
            formula = f"{self.target} ~ {' + '.join(self.features_to_scale)}"
            me_model = mixedlm(formula, train_df, groups=train_df[self.participant_id_col])
            self.mixed_effects_model_result = me_model.fit()
            logger.info("Global Mixed effects model fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting global mixed effects model: {str(e)}")

        # --- Random Forest Regressor ---
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state
            )
            rf_model.fit(X_train, y_train)
            self.global_alternative_models["random_forest"] = rf_model
            logger.info("Global Random Forest model fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting global Random Forest model: {str(e)}")

        # --- XGBoost Regressor ---
        try:
            xgb_model = XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, objective="reg:squarederror",
                enable_categorical=False # XGBoost handles categorical differently
            )
            xgb_model.fit(X_train, y_train)
            self.global_alternative_models["xgboost"] = xgb_model
            logger.info("Global XGBoost model fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting global XGBoost model: {str(e)}")

    def fine_tune_model_for_user(self, user_id, user_df, model_type="random_forest", min_samples=15):
        logger.info(f"Attempting to fine-tune {model_type} for user {user_id}")
        if len(user_df) < min_samples:
            logger.warning(
                f"Not enough data ({len(user_df)} points, need {min_samples}) for user {user_id} to fine-tune {model_type}. Skipping."
            )
            return False

        # user_df is already scaled from preprocess_data
        X_user = user_df[self.features_to_scale]
        y_user = user_df[self.target]

        user_model_key = f"{user_id}_{model_type}"
        user_model_instance = None

        try:
            if model_type == "random_forest":
                user_model_instance = RandomForestRegressor(
                    n_estimators=50, max_depth=7, random_state=self.random_state # Potentially different params
                )
            elif model_type == "xgboost":
                user_model_instance = XGBRegressor(
                    n_estimators=50, max_depth=5, learning_rate=0.1,
                    random_state=self.random_state, objective="reg:squarederror"
                )
            else:
                logger.error(f"Unsupported model_type '{model_type}' for fine-tuning.")
                return False

            user_model_instance.fit(X_user, y_user)
            self.user_specific_models[user_model_key] = user_model_instance

            # Store sample data for this user-specific model's explainer
            sample_size = min(max(1, len(X_user) // 2), 50) # Ensure at least 1, up to 50
            self.user_specific_explainer_data[user_model_key] = X_user.sample(
                n=sample_size, random_state=self.random_state
            )
            logger.info(f"Successfully fine-tuned and stored model and explainer data for {user_model_key}")
            return True
        except Exception as e:
            logger.error(f"Error fine-tuning {model_type} for user {user_id}: {e}")
            if user_model_key in self.user_specific_models:
                del self.user_specific_models[user_model_key]
            if user_model_key in self.user_specific_explainer_data:
                del self.user_specific_explainer_data[user_model_key]
            return False

    def evaluate_models(self, test_df, user_id=None):
        # If user_id is provided, this evaluates that specific user's model on their portion of test_df
        # Otherwise, evaluates global models on the whole test_df
        evaluation_results = {}
        if not all(f in test_df.columns for f in self.features_to_scale + [self.target]):
            logger.error("Test data missing required columns. Cannot evaluate.")
            return {}

        X_test = test_df[self.features_to_scale]
        y_test = test_df[self.target]

        if X_test.empty:
            logger.warning("X_test is empty. Cannot evaluate.")
            return {}

        models_to_evaluate = {}
        if user_id:
            for model_key, model_instance in self.user_specific_models.items():
                if model_key.startswith(f"{user_id}_"):
                    models_to_evaluate[model_key] = model_instance
            if not models_to_evaluate:
                logger.warning(f"No specific models found for user {user_id} to evaluate.")
                return {}
            logger.info(f"Evaluating specific models for user {user_id}...")
        else:
            models_to_evaluate = self.global_alternative_models
            logger.info("Evaluating global models...")
            if self.mixed_effects_model_result: # Add ME model for global evaluation
                try:
                    params = self.mixed_effects_model_result.params
                    intercept = params.get('Intercept', 0)
                    predictions_me = pd.Series(intercept, index=X_test.index)
                    for feature in self.features_to_scale:
                        if feature in params:
                             predictions_me += params[feature] * X_test[feature]
                    evaluation_results['mixed_effects (global)'] = {
                        'mse': mean_squared_error(y_test, predictions_me),
                        'rmse': np.sqrt(mean_squared_error(y_test, predictions_me)),
                        'r2': r2_score(y_test, predictions_me),
                        'mae': mean_absolute_error(y_test, predictions_me)
                    }
                except Exception as e:
                    logger.error(f"Error evaluating mixed effects model: {str(e)}")
                    evaluation_results['mixed_effects (global)'] = {'error': str(e)}


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
                logger.error(f"Error evaluating {name} model: {str(e)}")
                evaluation_results[name] = {"error": str(e)}

        for model_name, metrics in evaluation_results.items():
            if "error" not in metrics:
                logger.info(
                    f"{model_name} perf: RMSE={metrics.get('rmse', float('nan')):.3f}, R²={metrics.get('r2', float('nan')):.3f}, MAE={metrics.get('mae', float('nan')):.3f}"
                )
        return evaluation_results

    def cross_validate_global_models(self, df, n_splits=5):
        logger.info(f"Starting {n_splits}-fold cross-validation for global models...")
        # ... (implementation largely unchanged, operates on global models)
        # Ensure it uses self.global_alternative_models
        if not all(f in df.columns for f in self.features_to_scale + [self.target]):
            logger.error("Data for CV missing columns. Cannot perform CV.")
            return {}

        X = df[self.features_to_scale]
        y = df[self.target]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_results = {name: {'rmse': [], 'r2': []} for name in self.global_alternative_models.keys()}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.info(f"CV fold {fold+1}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if 'random_forest' in self.global_alternative_models:
                try:
                    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state)
                    rf.fit(X_train, y_train)
                    preds_rf = rf.predict(X_test)
                    cv_results['random_forest']['rmse'].append(np.sqrt(mean_squared_error(y_test, preds_rf)))
                    cv_results['random_forest']['r2'].append(r2_score(y_test, preds_rf))
                except Exception as e: logger.error(f"Error in RF CV fold {fold+1}: {e}")

            if 'xgboost' in self.global_alternative_models:
                try:
                    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                       random_state=self.random_state, objective='reg:squarederror')
                    xgb.fit(X_train, y_train)
                    preds_xgb = xgb.predict(X_test)
                    cv_results['xgboost']['rmse'].append(np.sqrt(mean_squared_error(y_test, preds_xgb)))
                    cv_results['xgboost']['r2'].append(r2_score(y_test, preds_xgb))
                except Exception as e: logger.error(f"Error in XGB CV fold {fold+1}: {e}")
        # ... (summary printing remains similar)
        summary = {}
        for model_name_cv in cv_results:
            if cv_results[model_name_cv]['rmse']: # Check if list is not empty
                summary[model_name_cv] = {
                    'mean_rmse': np.mean(cv_results[model_name_cv]['rmse']),
                    'std_rmse': np.std(cv_results[model_name_cv]['rmse']),
                    'mean_r2': np.mean(cv_results[model_name_cv]['r2']),
                    'std_r2': np.std(cv_results[model_name_cv]['r2'])
                }
                logger.info(f"{model_name_cv} CV: RMSE={summary[model_name_cv]['mean_rmse']:.3f}±{summary[model_name_cv]['std_rmse']:.3f}, R²={summary[model_name_cv]['mean_r2']:.3f}±{summary[model_name_cv]['std_r2']:.3f}")
            else:
                summary[model_name_cv] = {'error': 'No successful folds or model not in CV'}
        return summary

   

    def predict_sleep_quality(self, new_data_input, model_type="random_forest", user_id=None):
        if isinstance(new_data_input, dict):
            new_data_df = pd.DataFrame([new_data_input])
        elif isinstance(new_data_input, pd.DataFrame):
            new_data_df = new_data_input.copy()
        else:
            raise TypeError("Input data must be a dict or DataFrame.")

        # Ensure all features are present, add as NaN if missing, then select in order
        for feature in self.features_to_scale:
            if feature not in new_data_df.columns:
                new_data_df[feature] = np.nan # Will be imputed by scaler if it was fit on this feature
                                              # Or use a default like 0 if scaler wasn't fit on it.
                                              # Better: ensure input dict has all keys from self.features_to_scale

        # Impute NaNs that might have been introduced for missing columns before scaling
        # This assumes the scaler was fit on data where these features might have been present
        for feature in self.features_to_scale:
            if new_data_df[feature].isnull().any():
                # A simple imputation for prediction time; ideally, input data is complete
                # or use a more sophisticated imputation strategy consistent with training
                logger.warning(f"Feature {feature} has NaNs in input for prediction. Imputing with 0.")
                new_data_df[feature].fillna(0, inplace=True)


        # Scale the features
        new_data_df_ordered = new_data_df[self.features_to_scale]
        try:
            scaled_feature_values = self.scaler.transform(new_data_df_ordered)
        except Exception as e:
            logger.error(f"Error scaling input data for prediction: {e}. Ensure scaler is fitted and input has correct features.")
            raise
        scaled_features_df = pd.DataFrame(
            scaled_feature_values, columns=self.features_to_scale, index=new_data_df.index
        )

        model_to_use = None
        model_key_used = ""

        if user_id:
            user_model_key = f"{user_id}_{model_type}"
            if user_model_key in self.user_specific_models:
                model_to_use = self.user_specific_models[user_model_key]
                model_key_used = f"user-specific: {user_model_key}"
            else:
                logger.warning(
                    f"User-specific model {user_model_key} not found. Falling back to global {model_type}."
                )

        if model_to_use is None: # Fallback or no user_id
            if model_type in self.global_alternative_models:
                model_to_use = self.global_alternative_models[model_type]
                model_key_used = f"global: {model_type}"
            elif model_type == "mixed_effects" and self.mixed_effects_model_result:
                # Handle mixed effects prediction
                params = self.mixed_effects_model_result.params
                intercept = params.get('Intercept', 0)
                predictions = pd.Series(intercept, index=scaled_features_df.index)
                for feature in self.features_to_scale:
                    if feature in params:
                        predictions += params[feature] * scaled_features_df[feature]
                logger.info(f"Prediction using global: mixed_effects model")
                return float(predictions[0]) if len(predictions) == 1 else predictions.tolist()
            else:
                raise ValueError(f"Model {model_type} (global or user-specific for {user_id}) not available.")

        if model_to_use is None:
             raise ValueError(f"Could not find a suitable model for prediction (type: {model_type}, user: {user_id}).")

        logger.info(f"Prediction using model {model_key_used}")
        predictions = model_to_use.predict(scaled_features_df)
        return float(predictions[0]) if len(predictions) == 1 else predictions.tolist()


    def interpret_model(self, instance_df, model_type="random_forest", user_id=None):
        # Ensure instance_df is a DataFrame
        if isinstance(instance_df, dict):
            instance_df = pd.DataFrame([instance_df])
        elif not isinstance(instance_df, pd.DataFrame):
            raise TypeError("instance_df must be a dict or DataFrame.")

        # Ensure all features are present and in order, scale them
        for feature in self.features_to_scale:
            if feature not in instance_df.columns:
                instance_df[feature] = np.nan # Will be imputed
        for feature in self.features_to_scale: # Impute NaNs
            if instance_df[feature].isnull().any():
                logger.warning(f"Feature {feature} has NaNs in instance for interpretation. Imputing with 0.")
                instance_df[feature].fillna(0, inplace=True)

        instance_df_ordered = instance_df[self.features_to_scale]
        try:
            instance_scaled_values = self.scaler.transform(instance_df_ordered)
        except Exception as e:
            logger.error(f"Error scaling instance data for interpretation: {e}")
            return {"error": f"Scaling error: {e}"}

        instance_scaled_df = pd.DataFrame(
            instance_scaled_values, columns=self.features_to_scale, index=instance_df.index
        )
        # For LIME, it expects a 1D numpy array for a single instance
        instance_scaled_numpy = instance_scaled_values[0]


        model_to_explain = None
        background_data_for_explainer = None # This should be a DataFrame
        model_key_used = ""

        def lime_predict_fn_wrapper(numpy_data):
        # Convert numpy_data back to DataFrame with correct feature names
        # Assuming background_data_for_explainer has the correct column names
            df_for_predict = pd.DataFrame(numpy_data, columns=background_data_for_explainer.columns.tolist())
            return model_to_explain.predict(df_for_predict)

        if user_id:
            user_model_key = f"{user_id}_{model_type}"
            if user_model_key in self.user_specific_models:
                model_to_explain = self.user_specific_models[user_model_key]
                if user_model_key in self.user_specific_explainer_data:
                    background_data_for_explainer = self.user_specific_explainer_data[user_model_key]
                model_key_used = f"user-specific: {user_model_key}"
            else:
                logger.warning(f"User-specific model {user_model_key} not found for interpretation. Falling back.")

        if model_to_explain is None: # Fallback or no user_id
            if model_type in self.global_alternative_models:
                model_to_explain = self.global_alternative_models[model_type]
                background_data_for_explainer = self.global_explainer_data_sample
                model_key_used = f"global: {model_type}"
            # Add mixed_effects if you want to interpret it (though SHAP/LIME are less common for it)

        if model_to_explain is None:
            return {"error": f"Model {model_type} (global or user-specific for {user_id}) not found for interpretation."}
        if background_data_for_explainer is None or background_data_for_explainer.empty:
            logger.error(f"Background data for explainer of {model_key_used} is missing or empty.")
            return {"error": f"Background data for explainer of {model_key_used} missing."}

            # ... inside interpret_model, after model_to_explain and background_data_for_explainer are set ...

        logger.info(f"Interpreting model {model_key_used}")
        explanations = {'model_interpreted': model_key_used}

        # SHAP Explanations
        try:
            shap_explainer = shap.Explainer(model_to_explain, background_data_for_explainer)
            shap_values = shap_explainer.shap_values(instance_scaled_df)
            explanations["shap_values"] = shap_values[0].tolist() if isinstance(shap_values, np.ndarray) and shap_values.ndim > 0 else shap_values.tolist()
            explanations["shap_feature_names"] = background_data_for_explainer.columns.tolist()
        except Exception as e:
            logger.error(f"Error generating SHAP for {model_key_used}: {e}")
            explanations["shap_error"] = str(e)

        # LIME Explanations
        try:
            # Wrapper for the predict_fn for LIME
            def lime_predict_fn_wrapper(numpy_data):
                # Ensure numpy_data is 2D
                if numpy_data.ndim == 1:
                    numpy_data_2d = numpy_data.reshape(1, -1)
                else:
                    numpy_data_2d = numpy_data
                df_for_predict = pd.DataFrame(
                    numpy_data_2d,
                    columns=background_data_for_explainer.columns.tolist()
                )
                return model_to_explain.predict(df_for_predict)

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=background_data_for_explainer.values,
                feature_names=background_data_for_explainer.columns.tolist(),
                mode="regression",
                random_state=self.random_state,
            )
            lime_exp = lime_explainer.explain_instance(
                data_row=instance_scaled_numpy,
                predict_fn=lime_predict_fn_wrapper, # Use the wrapper
                num_features=len(self.features_to_scale),
            )
            explanations["lime_explanation"] = lime_exp.as_list()
        except Exception as e:
            logger.error(f"Error generating LIME for {model_key_used}: {e}")
            explanations["lime_error"] = str(e)

        return explanations


    def save_model(self, file_path="sleep_model_bundle.pkl"):
        model_data = {
            "random_state": self.random_state,
            "scaler": self.scaler,
            "mixed_effects_model_result": self.mixed_effects_model_result,
            "global_alternative_models": self.global_alternative_models,
            "user_specific_models": self.user_specific_models,
            "global_explainer_data_sample": self.global_explainer_data_sample, # DataFrame
            "user_specific_explainer_data": self.user_specific_explainer_data, # Dict of DataFrames
            "features_to_scale": self.features_to_scale,
            "target": self.target,
            "participant_id_col": self.participant_id_col,
            "date_col": self.date_col,
        }
        try:
            joblib.dump(model_data, file_path)
            logger.info(f"Model bundle saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model bundle: {str(e)}")
            raise

    @staticmethod
    def load_model(file_path="sleep_model_bundle.pkl"):
        try:
            model_data = joblib.load(file_path)
            model = SleepQualityModel(random_state=model_data.get("random_state", 42))

            model.scaler = model_data.get("scaler")
            model.mixed_effects_model_result = model_data.get("mixed_effects_model_result")
            model.global_alternative_models = model_data.get("global_alternative_models", {})
            model.user_specific_models = model_data.get("user_specific_models", {})
            model.global_explainer_data_sample = model_data.get("global_explainer_data_sample")
            model.user_specific_explainer_data = model_data.get("user_specific_explainer_data", {})
            model.features_to_scale = model_data.get("features_to_scale", [])
            model.target = model_data.get("target")
            model.participant_id_col = model_data.get("participant_id_col")
            model.date_col = model_data.get("date_col")

            if not model.features_to_scale:
                 logger.warning("Loaded model has no 'features_to_scale'. This might cause issues.")
            if model.scaler is None:
                 logger.warning("Loaded model has no 'scaler'. Predictions on new data will fail if scaling is needed.")


            logger.info(f"Model bundle loaded successfully from {file_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model bundle: {str(e)}")
            raise

    def create_api(self):
        app = Flask(__name__)
        # Make self available to the route handlers
        # This is a common pattern but be mindful of state if app scales
        app.config['SLEEP_MODEL_INSTANCE'] = self

        @app.route("/predict", methods=["POST", "OPTIONS"])
        def predict_route():
            model_instance = app.config['SLEEP_MODEL_INSTANCE']
            try:
                data_json = request.json
                if not data_json:
                    return jsonify({"error": "Request body must be JSON."}), 400

                instance_dict = data_json.get("instance")
                if not instance_dict or not isinstance(instance_dict, dict):
                    return jsonify({"error": "Missing 'instance' dictionary in payload."}), 400

                # Ensure all features are present for prediction
                for f_name in model_instance.features_to_scale:
                    if f_name not in instance_dict:
                        # Add a default or raise error. Here, adding a default (e.g., 0 or mean)
                        # This should align with how missing data is handled in preprocess_data
                        logger.warning(f"Feature '{f_name}' missing in API input, defaulting to 0 for prediction.")
                        instance_dict[f_name] = 0 # Or np.nan and let preprocess handle

                user_id = data_json.get("user_id") # Optional
                model_type_pref = data_json.get("model_type", "random_forest")

                prediction_value = model_instance.predict_sleep_quality(
                    instance_dict, model_type=model_type_pref, user_id=user_id
                )

                explanation_output = model_instance.interpret_model(
                    instance_dict, model_type=model_type_pref, user_id=user_id
                )

                return jsonify({
                    "prediction": prediction_value, # Already float or list from predict_sleep_quality
                    "explanation": explanation_output,
                    "model_used_for_prediction_details": explanation_output.get('model_interpreted', 'N/A'),
                    "input_features": instance_dict,
                    "status": "success",
                })
            except ValueError as ve:
                logger.error(f"API ValueError: {str(ve)}")
                return jsonify({"error": str(ve), "status": "error"}), 400
            except Exception as e:
                logger.error(f"API Exception: {e}", exc_info=True)
                return jsonify({"error": f"Unexpected error: {str(e)}", "status": "error"}), 500
        return app

def generate_synthetic_data_if_needed(file_path="synthetic_sleep_enhanced_v3.csv", num_rows=1000, num_participants=50):
    if os.path.exists(file_path):
        logger.info(f"Data file {file_path} already exists. Skipping generation.")
        return

    logger.info(f"Generating synthetic data at {file_path}...")
    np.random.seed(42)
    data = pd.DataFrame()
    data['ParticipantID'] = np.random.choice([f'User_{i}' for i in range(num_participants)], num_rows)
    base_date = datetime(2024, 1, 1)
    data['Date'] = [base_date + timedelta(days=int(i)) for i in np.random.randint(0, 365, num_rows)]
    data.sort_values(by=['ParticipantID', 'Date'], inplace=True)

    # Lifestyle & Environmental Factors
    data['ScreenTime'] = np.random.uniform(0, 6, num_rows)  # hours
    data['StressLevel'] = np.random.randint(1, 11, num_rows) # 1-10 scale
    data['DietScore'] = np.random.randint(1, 11, num_rows) # 1-10 healthy eating score
    data['PhysicalActivity'] = np.random.uniform(0, 120, num_rows) # minutes
    data['CaffeineIntake'] = np.random.choice([0, 50, 100, 150, 200], num_rows, p=[0.3, 0.3, 0.2, 0.1, 0.1]) # mg
    data['BedroomNoise'] = np.random.uniform(0, 1, num_rows) # 0=silent, 1=noisy
    data['BedroomLight'] = np.random.uniform(0, 1, num_rows) # 0=dark, 1=bright
    data['EveningAlcohol'] = np.random.choice([0, 1, 2, 3], num_rows, p=[0.7, 0.15, 0.1, 0.05]) # units
    data['ExerciseFrequency'] = np.random.randint(0, 8, num_rows) # days per week
    data['MindfulnessPractice'] = np.random.choice([0, 1], num_rows, p=[0.8, 0.2]) # 0=no, 1=yes

    # Derived & Stability Metrics
    data['CircadianStability'] = 1 - np.abs(np.random.normal(0, 0.2, num_rows)) # 0=unstable, 1=stable
    data['SocialJetlag'] = np.abs(np.random.normal(0, 1, num_rows)) # hours difference work/free days

    # Placeholder for Wearable Data (initially can be mostly NaNs or simple random)
    data['AvgHRV'] = np.random.uniform(20, 100, num_rows)
    data['DeepSleepProportion'] = np.random.uniform(0.05, 0.35, num_rows)
    data['StepsToday'] = np.random.randint(500, 20000, num_rows)
    data['RestingHR'] = np.random.randint(45, 90, num_rows)


    # Target: SleepQuality (1-10) - make it somewhat dependent on other factors
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

    data.to_csv(file_path, index=False)
    logger.info(f"Synthetic data generated and saved to {file_path}")


def main(data_file_path="synthetic_sleep_enhanced_v3.csv"):
    try:
        generate_synthetic_data_if_needed(data_file_path) # Generate if not exists

        sleep_model = SleepQualityModel(random_state=42)
        full_data = sleep_model.load_data(data_file_path)
        processed_data = sleep_model.preprocess_data(full_data.copy()) # Pass a copy

        # --- EDA and Visualization (Optional) ---
        # eda_results = sleep_model.exploratory_data_analysis(processed_data.copy()) # Pass copy
        # print("\nEDA Summary Stats (Head):\n", eda_results.get('summary_statistics', pd.DataFrame()).head())
        # sleep_model.visualize_data(processed_data.copy()) # Pass copy
        # plt.show() # Explicitly show plots if visualize_data is called

        # --- Split Data ---
        # Use participant-based split for training global models to avoid data leakage
        # Temporal split might be better if you have long series per user
        train_df, test_df = sleep_model.split_data(processed_data, test_size=0.25, temporal=False)

        if train_df.empty:
            logger.error("Training data is empty after split. Aborting.")
            return
        if test_df.empty:
            logger.warning("Test data is empty after split. Evaluation will be skipped.")

        # --- Fit Global Models ---
        sleep_model.fit_global_models(train_df.copy()) # Pass copy

        # --- Evaluate Global Models ---
        if not test_df.empty:
            logger.info("\n--- Evaluating Global Models ---")
            global_eval_results = sleep_model.evaluate_models(test_df.copy()) # Pass copy
            print("Global Model Evaluation Results:\n", global_eval_results)
        else:
            logger.info("Skipping global model evaluation as test_df is empty.")

        # --- Cross-Validate Global Models ---
        # logger.info("\n--- Cross-Validating Global Models ---")
        # cv_results = sleep_model.cross_validate_global_models(train_df.copy(), n_splits=3) # Use train_df or processed_data
        # print("Global Model Cross-Validation Results:\n", cv_results)


        # --- Simulate User-Specific Fine-Tuning ---
        logger.info("\n--- Simulating User-Specific Fine-Tuning ---")
        # Example: Fine-tune for a couple of users if they exist and have enough data
        unique_users_in_train = train_df[sleep_model.participant_id_col].unique()
        users_to_fine_tune = unique_users_in_train[:2] # Take first two for demo

        for user_id_to_tune in users_to_fine_tune:
            user_specific_training_data = train_df[
                train_df[sleep_model.participant_id_col] == user_id_to_tune
            ]
            if not user_specific_training_data.empty:
                sleep_model.fine_tune_model_for_user(
                    user_id=user_id_to_tune,
                    user_df=user_specific_training_data.copy(), # Pass copy
                    model_type="random_forest",
                    min_samples=5 # Lower min_samples for demo with potentially small user data
                )
                # Optionally, evaluate this user-specific model on their portion of test_df
                user_test_data = test_df[test_df[sleep_model.participant_id_col] == user_id_to_tune]
                if not user_test_data.empty:
                    logger.info(f"\n--- Evaluating Fine-Tuned Model for User: {user_id_to_tune} ---")
                    user_eval = sleep_model.evaluate_models(user_test_data.copy(), user_id=user_id_to_tune)
                    print(f"Evaluation for {user_id_to_tune}:\n", user_eval)
            else:
                logger.info(f"User {user_id_to_tune} has no data in training set for fine-tuning.")


        # --- Save and Load Model Bundle ---
        logger.info("\n--- Saving and Loading Model Bundle ---")
        sleep_model.save_model(file_path="sleep_model_bundle_v2.pkl")
        loaded_model = SleepQualityModel.load_model(file_path="sleep_model_bundle_v2.pkl")

        if loaded_model is None:
            logger.error("Failed to load model. Aborting further tests.")
            return

        # --- Test Prediction and Interpretation with Loaded Model ---
        logger.info("\n--- Testing Predictions with Loaded Model ---")

        # 1. Test with a global model prediction
        if not test_df.empty:
            sample_global_instance_dict = test_df.iloc[[0]][loaded_model.features_to_scale].to_dict(orient='records')[0]
            # Add ParticipantID and SleepQuality if needed by some internal logic, though predict_sleep_quality primarily uses features_to_scale
            sample_global_instance_dict[loaded_model.target] = test_df.iloc[0][loaded_model.target]
            sample_global_instance_dict[loaded_model.participant_id_col] = test_df.iloc[0][loaded_model.participant_id_col]


            global_pred = loaded_model.predict_sleep_quality(
                sample_global_instance_dict, model_type="random_forest" # No user_id
            )
            print(f"Global Prediction for sample: {global_pred:.2f} (Actual: {sample_global_instance_dict[loaded_model.target]})")
            global_interp = loaded_model.interpret_model(
                sample_global_instance_dict, model_type="random_forest" # No user_id
            )
            print("Global Interpretation (LIME sample):")
            if global_interp.get('lime_explanation'):
                for feature, weight in global_interp['lime_explanation'][:3]: print(f"  {feature}: {weight:.3f}")
            else: print(f"  LIME Error: {global_interp.get('lime_error', 'Not available')}")

        user_to_test_pred = users_to_fine_tune[0] # Or iterate through users_to_fine_tune
        user_model_key_to_check = f"{user_to_test_pred}_random_forest" # Assuming RF was tuned


        # 2. Test with a user-specific model prediction (if one was fine-tuned and loaded)
        if users_to_fine_tune: # Check if the list is not empty
            user_to_test_pred = users_to_fine_tune[0] # Example: test the first user fine-tuned
            user_model_key_to_check = f"{user_to_test_pred}_random_forest" # Assuming RF was the type fine-tuned

            if user_model_key_to_check in loaded_model.user_specific_models:
                logger.info(f"User-specific model '{user_model_key_to_check}' found in loaded bundle.")
                user_sample_df_test = test_df[test_df[loaded_model.participant_id_col] == user_to_test_pred]

                if not user_sample_df_test.empty:
                    # Prepare a sample instance from the test set for this user
                    # Ensure all features_to_scale are present in the dict
                    sample_user_instance_dict = {}
                    for f_col in loaded_model.features_to_scale:
                        if f_col in user_sample_df_test.columns:
                            sample_user_instance_dict[f_col] = user_sample_df_test.iloc[0][f_col]
                        else:
                            sample_user_instance_dict[f_col] = 0 # Or np.nan, consistent with preprocessing

                    # Add target and ID for logging/comparison, not strictly for prediction input
                    sample_user_instance_dict[loaded_model.target] = user_sample_df_test.iloc[0][loaded_model.target]
                    sample_user_instance_dict[loaded_model.participant_id_col] = user_to_test_pred

                    user_pred = loaded_model.predict_sleep_quality(
                        sample_user_instance_dict, model_type="random_forest", user_id=user_to_test_pred
                    )
                    print(f"\nUser-Specific Prediction for {user_to_test_pred}: {user_pred:.2f} (Actual: {sample_user_instance_dict[loaded_model.target]})")

                    user_interp = loaded_model.interpret_model(
                        sample_user_instance_dict, model_type="random_forest", user_id=user_to_test_pred
                    )
                    print(f"User-Specific Interpretation for {user_to_test_pred} (LIME sample):")
                    if user_interp.get('lime_explanation'):
                        for feature, weight in user_interp['lime_explanation'][:3]: print(f"  {feature}: {weight:.3f}")
                    elif 'lime_error' in user_interp: print(f"  LIME Error: {user_interp['lime_error']}")
                    else: print("  LIME explanation not available or in unexpected format.")
                else:
                    logger.info(f"No test data found for user {user_to_test_pred} to test their specific prediction.")
            else:
                logger.info(f"User-specific model for {user_to_test_pred} (key: {user_model_key_to_check}) not found in loaded_model.user_specific_models.")
        else:
            logger.info("No users were designated for fine-tuning in this run (users_to_fine_tune list is empty).")



        # --- API (Example Usage - not run by default) ---
        logger.info("\n--- API Setup ---")
        # api_app = loaded_model.create_api()
        logger.info("Flask API object created. To run (example): api_app.run(host='0.0.0.0', port=5001, debug=False)")
        # Example API call (using requests library, if you were to run the API separately):
        import requests
        payload = {
            "user_id": users_to_fine_tune[0] if users_to_fine_tune else None, # Example user
            "model_type": "random_forest",
            "instance": sample_user_instance_dict if users_to_fine_tune and not user_sample_df_test.empty else sample_global_instance_dict
        }
        try:
            response = requests.post("http://localhost:5001/predict", json=payload)
            print("\nAPI Response:", response.json())
        except Exception as api_e:
            print(f"API call failed (ensure API is running): {api_e}")


    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file_path}.")
    except ValueError as ve:
        logger.error(f"ValueError in main workflow: {str(ve)}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in main workflow: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

# --- Flask App Exposure for Render ---
try:
    model = SleepQualityModel.load_model(file_path="sleep_model_bundle_v2.pkl")
    app = model.create_api()
except Exception as e:
    import logging
    from flask import Flask, jsonify
    logging.error(f"Failed to load model or create API: {e}", exc_info=True)
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins (for testing)

    @app.route("/")
    def fallback_route():
        return jsonify({"error": "Failed to initialize model.", "details": str(e)})
