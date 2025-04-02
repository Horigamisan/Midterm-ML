import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def preprocess_data(X, y, test_size=0.25, random_state=42):
    
    print("\n--- Starting Data Preprocessing (Scaling, Encoding, Splitting) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y# Stratify for classification
    )

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    numeric_pipeline = Pipeline([
        ('scaler',RobustScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    try:
        # Get feature names from the fitted transformer
        feature_names_out = preprocessor.get_feature_names_out()
        print(f"  Number of features after preprocessing: {len(feature_names_out)}")

        # Convert processed arrays back to DataFrames with proper column names
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)

    except Exception as e:
        print(f"Warning: Could not retrieve feature names automatically. Error: {e}")
        print("Proceeding with NumPy arrays. Feature importance plots might lack labels.")
        # Keep as numpy arrays if get_feature_names_out fails (older sklearn versions?)

    print(f"\nPreprocessing finished.")
    print(f"  Processed Train set shape: X={X_train_processed.shape}, y={y_train.shape}")
    print(f"  Processed Test set shape:  X={X_test_processed.shape}, y={y_test.shape}")
    print("--- End Data Preprocessing ---")


    return X_train_processed, X_test_processed, y_train, y_test