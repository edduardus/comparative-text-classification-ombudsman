# Carrega Bibliotecas
import pandas as pd
import torch
from utils.utils import prepara_dados, TextDataset
from utils.config import TextLSTM
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
import time
import os

# Função para carregar o modelo e fazer inferência
def load_and_infer(model, df_val, device='cuda'):
    start_time = time.time()  # Tempo de início

    device = torch.device(device)
    X_val, y_val = prepara_dados(df_val)
    val_dataset = TextDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular métricas
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    end_time = time.time()  # Tempo de fim
    elapsed_time = end_time - start_time  # Tempo decorrido

    print(f"F1 Score: {f1:.4f}")
    print(f"Kappa de Cohen: {kappa:.4f}")
    print(f"Tempo decorrido: {elapsed_time:.2f} segundos")

    return f1, kappa, elapsed_time

def main():
    print("Início do Processamento")
    
    # Carregar o modelo salvo referente ao fold 1
    best_model_fold = "best_lstm_model_fold_1.pt"
    best_model_path = os.path.join("modelos", "modelos_LSTM", best_model_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = TextLSTM(input_dim=768, hidden_dim=128, num_classes=6).to(device)  # Ajuste input_dim conforme necessário
    checkpoint = torch.load(best_model_path)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()

    resultados = []

    # Looping para inferência em todas as 30 bases distintas
    for i in range(1, 31):
        path_data_teste = os.path.join("data", "data_cross_validation", "amostra_teste", f"amostra_bootstrap_{i}.parquet")
        df_teste = pd.read_parquet(path_data_teste)
                
        print(f"### Fazendo inferência para a amostra {i} ###")
        f1, kappa, elapsed_time = load_and_infer(lstm_model, df_teste, device)
        resultados.append({'amostra': i, 'f1_score': f1, 'kappa': kappa, 'tempo': elapsed_time})

    # Armazenar os resultados em um DataFrame
    resultados_df = pd.DataFrame(resultados)
    path_avaliacao = os.path.join("data", 'evaluation_data', 'evaluation_data_lstm.parquet')
    resultados_df.to_parquet(path_avaliacao, index=False)
    print("Fim do Processamento")

if __name__ == "__main__":
    main()