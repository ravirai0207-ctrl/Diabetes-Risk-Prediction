import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Set up logging so we do not need to rely on print statments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClinicalRiskModeling")

def validate_clinical_schema(df: pd.DataFrame) -> bool:
    """Check to make sure the data is in right columns."""
    required_cols = {'ID', 'Glucose', 'BMI', 'Age', 'Outcome'}
    
    if not required_cols.issubset(df.columns):
        logger.error(f"Missing  columns: {required_cols - set(df.columns)}")
        return False
    return True

def preprocess_patient_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the raw data"""
    processed_df = df.copy()
    
    # export missing as exactly 0. 
    # no one had a BMI or BP 0 isliye drop karta hun.
    for vital in ['Glucose', 'BMI', 'BloodPressure']:
        if vital in processed_df.columns:
            processed_df = processed_df[processed_df[vital] != 0]

    
    if 'Location' in processed_df.columns:
        processed_df = pd.get_dummies(processed_df, columns=['Location'], dtype='int64')

    #  drop ID (by errors='ignore')
    return processed_df.drop(columns=['ID'], errors='ignore')

def train_diabetes_risk_baseline(data_path="diabetes2.csv", model_output_dir="./artifacts"):
    """Extract the data, trains a model and save it out."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Can not find data at {data_path}. Did you put it in the right folder?")

    try:
        logger.info("Grabbing patient data...")
        patient_vitals_df = pd.read_csv(data_path)
        
        # Stop everything if the data looks wrong
        if not validate_clinical_schema(patient_vitals_df):
            raise ValueError("Schema validation failed. Halting pipeline.")

        clean_cohort_df = preprocess_patient_vitals(patient_vitals_df)

        clinical_features = clean_cohort_df.drop(columns=['Outcome'])
        risk_outcome = clean_cohort_df['Outcome']

        # 70/30 split
        X_train, X_test, y_train, y_test = train_test_split(
            clinical_features, risk_outcome, test_size=0.30, random_state=42, stratify=risk_outcome
        )

       
        logger.info("Fitting the baseline model...")
        risk_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        risk_model.fit(X_train, y_train)

        predictions = risk_model.predict(X_test)
        predictive_probabilities = risk_model.predict_proba(X_test)[:, 1]
        
        # out the metric 
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, predictive_probabilities):.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, predictions)}")

        # Save the trained model 
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = os.path.join(model_output_dir, "baseline_diabetes_risk_model.pkl")
        joblib.dump(risk_model, model_path)
        logger.info(f"Model saved to {model_path}")

        return risk_model

    except Exception as e:
        logger.error("Something went wrong", exc_info=True)
        raise

if __name__ == "__main__":
    # Run the progam
    model = train_diabetes_risk_baseline(data_path="diabetes2.csv")