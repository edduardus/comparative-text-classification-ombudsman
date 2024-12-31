# Importa biliotecas
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import f1_score
import os
from utils.utils import prepara_dados
from utils.config import label2id
import time

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NUM_LABELS = len(label2id)
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Carregar o tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Função para tokenizar os textos
def tokenize_data(texts, labels):
    tokens = tokenizer(
        list(texts),  # Converter para lista
        padding=True,
        truncation=True,
        max_length=512,  # Ajuste conforme necessário
        return_tensors="pt"  # Retorna tensores do PyTorch
    )
    return tokens, torch.tensor(labels.tolist())  # Converter para lista antes do tensor

# Função para criar DataLoader
def create_dataloader(df):
    df['label'] = df['DescTipoManifestacao'].map(label2id)
    texts = df['TxtFatoManifestacao']
    labels = df['label']
    tokens, labels = tokenize_data(texts, labels)
    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

# Função para treinar o modelo
def train_model(train_loader, test_loader, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, torch_dtype=torch.bfloat16)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_f1 = 0.0
    pasta_modelo = os.path.join("modelos", "modelos_BERT")
    if not os.path.exists(pasta_modelo):
        os.makedirs(pasta_modelo)
    best_model_path = os.path.join(pasta_modelo, f"best_bert_model_fold_{fold}.pt")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        f1 = evaluate_model(model, test_loader, device)

        checkpoint_path = os.path.join(pasta_modelo, f"checkpoint_fold{fold}_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint salvo: {checkpoint_path}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Melhor modelo salvo com f1: {best_f1:.4f}")
    return best_f1

# Função para avaliar o modelo
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'F1 Score: {f1}')
    return f1






# Loop para carregar os dados, treinar e salvar os modelos para cada fold
print("Início do Processamento.")
path_data = os.path.join("data", "data_cross_validation")
path_tempos = os.path.join("data", 'training_data', 'training_data_bert.parquet')
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
    path_dados_terino = os.path.join(path_data, f'df_treino_fold_{fold + 1}.parquet')
    path_dados_val = os.path.join(path_data, f'df_val_fold_{fold + 1}.parquet')
    df_treino = pd.read_parquet(path_dados_terino)
    df_validacao = pd.read_parquet(path_dados_val)
    
    print(f"### Treinando modelo para o fold {fold + 1} ###")
    train_loader = create_dataloader(df_treino)
    test_loader = create_dataloader(df_validacao)

    f1 = train_model(train_loader, test_loader, fold + 1)
    
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