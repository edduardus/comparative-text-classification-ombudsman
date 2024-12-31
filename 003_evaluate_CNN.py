# Import Libraries
import pandas as pd
import torch
from utils.utils import prepara_dados, TextDataset
from utils.config import TextCNN
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
import time
import os

# Function to load the model and perform inference
def load_and_infer(model, df_val, device='cuda', batch_size=64):  # Increase batch_size
    start_time = time.time()  # Start time

    device = torch.device(device)
    
    # Prepare data
    data_preparation_start_time = time.time()
    X_val, y_val = prepara_dados(df_val)
    val_dataset = TextDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Use the new batch_size
    data_preparation_end_time = time.time()
    data_preparation_time = data_preparation_end_time - data_preparation_start_time
    
    # Perform inference
    inference_start_time = time.time()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Elapsed time

    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Data preparation time: {data_preparation_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    return f1, kappa, data_preparation_time, inference_time, elapsed_time

def main():
    print("Processing started")
    
    # Load the saved model for fold 1
    best_model_fold = "best_cnn_model_fold_1.pt"
    best_model_path = os.path.join("modelos", "modelos_CNN", best_model_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cnn_model = TextCNN(input_dim=768, num_classes=6).to(device)  # Adjust input_dim as needed
    checkpoint = torch.load(best_model_path)
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    cnn_model.eval()

    resultados = []

    # Loop for inference on all 30 different datasets
    for i in range(1, 31):
        path_data_teste = os.path.join("data", "data_cross_validation", "amostra_teste", f"amostra_bootstrap_{i}.parquet")
        df_teste = pd.read_parquet(path_data_teste)
        
        print(f"### Performing inference for sample {i} ###")
        f1, kappa, data_preparation_time, inference_time, elapsed_time = load_and_infer(cnn_model, df_teste, device, batch_size=64)  
        resultados.append({'sample': i, 'f1_score': f1, 'kappa': kappa, 'data_preparation_time': data_preparation_time, 'inference_time': inference_time, 'total_time': elapsed_time})

    # Store the results in a DataFrame
    resultados_df = pd.DataFrame(resultados)
    path_avaliacao = os.path.join("data", 'evaluation_data', 'evaluation_data_cnn.parquet')
    resultados_df.to_parquet(path_avaliacao, index=False)
    print("Processing completed")

if __name__ == "__main__":
    main()