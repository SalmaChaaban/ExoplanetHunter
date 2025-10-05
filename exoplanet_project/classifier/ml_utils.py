import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
import time
import os

NUMERIC_FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
]

MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(dataset_path, training_obj):
    """Train the exoplanet classifier"""
    start_time = time.time()
    
    # Load data
    df = pd.read_csv(dataset_path, comment='#')
    
    # Remove high-missing columns
    missing_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percent[missing_percent > 30].index
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Separate candidates
    candidates = df[df['koi_disposition'] == 'CANDIDATE']
    df = df[df['koi_disposition'] != 'CANDIDATE']
    
    # Balance dataset
    df_majority = df[df['koi_disposition'] == 'FALSE POSITIVE']
    df_minority = df[df['koi_disposition'] == 'CONFIRMED']
    
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )
    
    df_balanced = pd.concat([df_minority, df_majority_downsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Prepare features
    available_features = [f for f in NUMERIC_FEATURES if f in df_balanced.columns]
    X = df_balanced[available_features]
    y = df_balanced['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=available_features,
        index=X.index
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        stratify=y,
        test_size=training_obj.test_size,
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=training_obj.rf_n_estimators,
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=training_obj.gb_n_estimators,
            learning_rate=training_obj.gb_learning_rate,
            random_state=42
        ))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    # Train
    stack.fit(X_train_scaled, y_train)
    
    # Calibrate
    calibrated_stack = CalibratedClassifierCV(stack, cv=5, method='sigmoid')
    calibrated_stack.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = calibrated_stack.predict(X_test_scaled)
    y_proba = calibrated_stack.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    # Save model
    model_id = training_obj.id
    joblib.dump(calibrated_stack, os.path.join(MODEL_DIR, f'model_{model_id}.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{model_id}.pkl'))
    joblib.dump(imputer, os.path.join(MODEL_DIR, f'imputer_{model_id}.pkl'))
    
    with open(os.path.join(MODEL_DIR, f'features_{model_id}.txt'), 'w') as f:
        f.write('\n'.join(available_features))
    
    return {
    'n_samples': len(X),
    'n_features': len(available_features),
    'accuracy': accuracy * 100,  # Convert to percentage
    'f1_score': f1 * 100,
    'roc_auc': roc_auc * 100,
    'balanced_accuracy': balanced_acc * 100,
    'training_time': training_time,
}


def predict_candidates(candidates_path, training_obj):
    """Make predictions on candidate data"""
    # Load model
    model_id = training_obj.id
    model = joblib.load(os.path.join(MODEL_DIR, f'model_{model_id}.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, f'scaler_{model_id}.pkl'))
    imputer = joblib.load(os.path.join(MODEL_DIR, f'imputer_{model_id}.pkl'))
    
    with open(os.path.join(MODEL_DIR, f'features_{model_id}.txt'), 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    # Load candidates
    df = pd.read_csv(candidates_path, comment='#')
    
    # Get ID column
    id_col = None
    for col in ['kepid', 'koi_name', 'kepoi_name']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        df['temp_id'] = range(len(df))
        id_col = 'temp_id'
    
    # Prepare features
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    
    # Impute and scale
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    probas = model.predict_proba(X_scaled)[:, 1]
    classes = (probas > 0.5).astype(int)
    
    predictions = []
    for i, (koi_id, prob, cls) in enumerate(zip(df[id_col], probas, classes)):
        predictions.append({
            'koi_id': str(koi_id),
            'confidence': float(prob) * 100,  # Convert to percentage
            'class': 'CONFIRMED' if cls == 1 else 'FALSE POSITIVE'
        })
    
    return predictions
