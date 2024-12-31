# Import libraries
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from utils.utils import prepara_dados
import pandas as pd
import numpy as np
import os
import time
import joblib

# Function to train and evaluate models (per fold)
def train_and_evaluate(fold, df_train, df_val, num_classes=6, num_epochs=5):
    X_train, y_train = prepara_dados(df=df_train)
    X_val, y_val = prepara_dados(df=df_val)

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    all_preds = rf_model.predict(X_val)
    current_f1_score = f1_score(y_val, all_preds, average='weighted')
    print(f"F1 Score for fold {fold}: {current_f1_score:.4f}")

    # Save the best model
    best_model_fold = f"best_rf_model_fold_{fold}.joblib"
    best_model_path = os.path.join("modelos", "modelos_RF", best_model_fold)
    joblib.dump(rf_model, best_model_path)
    print(f"Best model saved with F1 Score: {current_f1_score:.4f}")

    return current_f1_score

def main():
    # Loop to load data, train and save models for each fold
    print("Processing started.")
    path_data = os.path.join("data", "data_cross_validation")
    path_tempos = os.path.join("data", 'training_data', 'training_data_rf.parquet')
    # List to store the times of each iteration
    tempos = []
    f1_scores = []

    # Check if the times file already exists
    if os.path.exists(path_tempos):
        tempos_df = pd.read_parquet(path_tempos)
        tempos = tempos_df['training_time'].tolist()
        f1_scores = tempos_df['f1_score_val'].tolist()
        start_fold = len(tempos)  # Continue from the next fold
    else:
        start_fold = 0

    for fold in range(start_fold, 1):
        start_time = time.time()  # Start time

        df_train = pd.read_parquet(os.path.join(path_data, f"df_treino_fold_{fold+1}.parquet"))
        df_val = pd.read_parquet(os.path.join(path_data, f"df_val_fold_{fold+1}.parquet"))

        print(f"### Starting training for fold {fold+1} ###")
        print(f"df_train.shape: {df_train.shape}, df_val.shape: {df_val.shape}")

        f1 = train_and_evaluate(fold+1, df_train, df_val)

        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Elapsed time
        tempos.append(elapsed_time)
        f1_scores.append(f1)
        print(f"Time for fold {fold+1}: {elapsed_time:.2f} seconds")
        print(f"F1 Score for fold {fold+1}: {f1:.4f}")

        # Save the times in a parquet file at each iteration
        tempos_df = pd.DataFrame({'fold': range(1, len(tempos) + 1), 'training_time': tempos, 'f1_score_val': f1_scores})
        tempos_df.to_parquet(path_tempos, index=False)
        print(f"Times saved in {path_tempos}")

    print("Processing completed.")

if __name__ == "__main__":
    main()