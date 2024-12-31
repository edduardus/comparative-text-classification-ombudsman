# Importa biliotecas
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from utils.config import TextCNN
from utils.utils import TextDataset, prepara_dados
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import time



# Função para treinar e avaliar os modelos (por fold)
def train_and_evaluate(fold, df_train, df_val, batch_size = 16, num_classes = 6, num_epochs = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train = prepara_dados(df=df_train)
    X_val, y_val = prepara_dados(df=df_val)
      
    
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    

    cnn_model = TextCNN(input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

    
    best_f1_score = 0.0
    best_model_fold = f"best_cnn_model_fold_{fold}.pt"
    best_model_path = os.path.join("modelos","modelos_CNN", best_model_fold)

    for epoch in range(num_epochs):
        cnn_model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            outputs = cnn_model(embeddings)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Época [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

        cnn_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = cnn_model(embeddings)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        current_f1_score = f1_score(all_labels, all_preds, average='weighted')
        print(f"F1 Score na época {epoch+1}: {current_f1_score:.4f}")

        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            torch.save({
                'model_state_dict': cnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_model_path)
            print(f"Novo melhor modelo salvo com F1 Score: {best_f1_score:.4f}")

    return best_f1_score


# Loop para carregar os dados, treinar e salvar os modelos para cada fold
print("Início do Processamento.")
path_data = os.path.join("data", "data_cross_validation")
path_tempos = os.path.join("data", 'training_data', 'training_data_cnn.parquet')
# Lista para armazenar os tempos de cada iteração
tempos = []
f1_scores = []

# Verificar se o arquivo de tempos já existe
if os.path.exists(path_tempos):
    tempos_df = pd.read_parquet(path_tempos)
    tempos = tempos_df['training_time'].tolist()
    f1_scores = tempos_df['f1_score_val'].tolist()
    start_fold = len(tempos)  # Continuar do próximo fold
else:
    start_fold = 0

for fold in range(start_fold, 1):
    start_time = time.time()  # Tempo de início

    df_train = pd.read_parquet(os.path.join(path_data, f"df_treino_fold_{fold+1}.parquet"))
    df_val = pd.read_parquet(os.path.join(path_data, f"df_val_fold_{fold+1}.parquet"))
    
    print(f"### Iniciando treinamento do fold {fold+1} ###")
    print(f"df_train.shape: {df_train.shape}, df_val.shape: {df_val.shape}")

    f1 = train_and_evaluate(fold+1, df_train, df_val)
    
    end_time = time.time()  # Tempo de fim
    elapsed_time = end_time - start_time  # Tempo decorrido
    tempos.append(elapsed_time)
    f1_scores.append(f1)
    print(f"Tempo para o fold {fold+1}: {elapsed_time:.2f} segundos")
    print(f"F1 Score para o fold {fold+1}: {f1:.4f}")

    # Salvar os tempos em um arquivo parquet a cada iteração
    tempos_df = pd.DataFrame({'fold': range(1, len(tempos) + 1), 'training_time': tempos, 'f1_score_val': f1_scores})
    tempos_df.to_parquet(path_tempos, index=False)
    print(f"Tempos salvos em {path_tempos}")

print("Processamento concluído.")