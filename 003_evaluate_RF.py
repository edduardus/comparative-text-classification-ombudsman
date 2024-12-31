# Load Libraries
import pandas as pd
import joblib
from utils.utils import prepara_dados
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
import time
import os

# Function to load the model and perform inference
def load_and_infer(model, df_val):
    start_time = time.time()  # Start time

    X_val, y_val = prepara_dados(df_val)

    # Make predictions
    all_preds = model.predict(X_val)
    
    # Calculate metrics
    f1 = f1_score(y_val, all_preds, average='weighted')
    kappa = cohen_kappa_score(y_val, all_preds)
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Elapsed time

    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return f1, kappa, elapsed_time

def main():
    print("Processing started")
    
    # Load the saved model for fold 1
    best_model_fold = "best_rf_model_fold_1.joblib"
    best_model_path = os.path.join("modelos", "modelos_RF", best_model_fold)
    rf_model = joblib.load(best_model_path)

    resultados = []

    # Loop for inference on all 30 different datasets
    for i in range(1, 31):
        path_data_teste = os.path.join("data", "data_cross_validation", "amostra_teste", f"amostra_bootstrap_{i}.parquet")
        df_teste = pd.read_parquet(path_data_teste)
                
        print(f"### Performing inference for sample {i} ###")
        f1, kappa, elapsed_time = load_and_infer(rf_model, df_teste)
        resultados.append({'sample': i, 'f1_score': f1, 'kappa': kappa, 'time': elapsed_time})

    # Store the results in a DataFrame
    resultados_df = pd.DataFrame(resultados)
    path_avaliacao = os.path.join("data", 'evaluation_data', 'evaluation_data_rf.parquet')
    resultados_df.to_parquet(path_avaliacao, index=False)
    print("Processing completed")

if __name__ == "__main__":
    main()