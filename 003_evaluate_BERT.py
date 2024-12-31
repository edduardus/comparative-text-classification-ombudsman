# Carrega Bibliotecas
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from utils.config import label2id, id2label
import time
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score

NUM_LABELS = len(label2id)
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
PASTA_MODELOS = os.path.join("modelos", "modelos_BERT")

# Função para tokenizar os textos
def tokenize_data(texts):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,  # Ajuste conforme necessário
        return_tensors="pt"  # Retorna tensores do PyTorch
    )
    return tokens

# Função para carregar o modelo e fazer inferência
def load_and_infer(model, df, device='cuda', batch_size=64):  # Adicionar batch_size como argumento
    start_time = time.time()  # Tempo de início

    df['texto'] = df['TxtFatoManifestacao']
    df['label'] = df['DescTipoManifestacao']
    df['label'] = df['label'].map(label2id)
    
    model.to(device)
    
    # Tokenizar os dados do DataFrame
    tokenization_start_time = time.time()
    tokens = tokenize_data(df["texto"])
    tokenization_end_time = time.time()
    tokenization_time = tokenization_end_time - tokenization_start_time

    # Converter os rótulos em tensores
    labels = torch.tensor(df["label"].values)

    # Criar TensorDataset
    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)

    # Criar DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Função para fazer previsões
    def predict():
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    # Fazer previsões
    inference_start_time = time.time()
    predictions = predict()
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Converter os resultados de números para strings
    predicted_labels = [id2label[pred] for pred in predictions]

    # Adicionar as previsões ao DataFrame
    df["predicted_label"] = predicted_labels
    
    # Calcular métricas
    f1 = f1_score(df['DescTipoManifestacao'], df['predicted_label'], average='weighted')
    kappa = cohen_kappa_score(df['DescTipoManifestacao'], df['predicted_label'])
    end_time = time.time()  # Tempo de fim
    elapsed_time = end_time - start_time  # Tempo decorrido

    print(f"F1 Score: {f1:.4f}")
    print(f"Kappa de Cohen: {kappa:.4f}")
    print(f"Tempo de tokenização: {tokenization_time:.2f} segundos")
    print(f"Tempo de inferência: {inference_time:.2f} segundos")
    print(f"Tempo total decorrido: {elapsed_time:.2f} segundos")

    return f1, kappa, tokenization_time, inference_time, elapsed_time

def main():
    print("Início do Processamento")
    
    # Carregar o modelo salvo referente ao fold 1
    best_model_path = os.path.join(PASTA_MODELOS, 'best_bert_model_fold_1.pt')
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, torch_dtype=torch.bfloat16)
    model.load_state_dict(torch.load(best_model_path))

    resultados = []

    # Looping para inferência em todas as 30 bases distintas
    for i in range(1, 31):
        path_data_teste = os.path.join("data", "data_cross_validation", "amostra_teste", f"amostra_bootstrap_{i}.parquet")
        df_teste = pd.read_parquet(path_data_teste)
                
        print(f"### Fazendo inferência para a amostra {i} ###")
        f1, kappa, tokenization_time, inference_time, elapsed_time = load_and_infer(model, df_teste, batch_size=128)
        resultados.append({'amostra': i, 'f1_score': f1, 'kappa': kappa, 'tempo_tokenizacao': tokenization_time, 'tempo_inferencia': inference_time, 'tempo_total': elapsed_time})

    # Armazenar os resultados em um DataFrame
    resultados_df = pd.DataFrame(resultados)
    path_avaliacao = os.path.join("data", 'evaluation_data', 'evaluation_data_bert.parquet')
    resultados_df.to_parquet(path_avaliacao, index=False)
    print("Fim do Processamento")

if __name__ == "__main__":
    main()