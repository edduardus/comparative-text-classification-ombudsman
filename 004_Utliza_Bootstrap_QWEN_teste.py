import pandas as pd
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from utils.config import label2id, id2label
import time

def load_model_and_tokenizer(model_path):
    #model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(id2label), id2label=id2label, label2id=label2id, use_cache=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(label2id), id2label=id2label, 
        label2id=label2id, use_cache=False, 
        torch_dtype=torch.bfloat16, device_map='auto',
        #attn_implementation="flash_attention_2"        
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Definir um token de preenchimento se não estiver definido
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def preprocess_function(texts, tokenizer):
    return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

def predict(model, tokenizer, df):
    start_time = time.time()  # Tempo de início
    texts = df["TxtFatoManifestacao"].tolist()
    predicted_labels = []

    model.eval()
    with torch.no_grad():
        for text in texts:
            tokenized_text = preprocess_function(text, tokenizer)
            tokenized_text.to(model.device)
            outputs = model(**tokenized_text)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            predicted_labels.append(id2label[prediction.item()])

    df["classe_Qwen"] = predicted_labels
    # Calcular métricas
    f1 = f1_score(df["DescTipoManifestacao"].values, predicted_labels, average='weighted')
    kappa = cohen_kappa_score(df["DescTipoManifestacao"].values, predicted_labels)
    end_time = time.time()  # Tempo de fim
    elapsed_time = end_time - start_time  # Tempo decorrido

    print(f"F1 Score: {f1:.4f}")
    print(f"Kappa de Cohen: {kappa:.4f}")
    print(f"Tempo decorrido: {elapsed_time:.2f} segundos")
    return f1, kappa, elapsed_time

def main():
    print("Início do Processamento")
    
    # Carregar o modelo salvo referente ao fold 1
    BASE_PATH_OUTPUT_MODEL = os.path.join('modelos', 'modelos_QWEN_old')
    model_name = os.path.join(BASE_PATH_OUTPUT_MODEL, "qwen_model_fold_1", "best_model")
    model, tokenizer = load_model_and_tokenizer(model_name)

    resultados = []

    # Looping para inferência em todas as 30 bases distintas
    for i in range(1, 31):
        path_data_teste = os.path.join("data", "data_cross_validation", "amostra_teste", f"amostra_bootstrap_{i}.parquet")
        df_teste = pd.read_parquet(path_data_teste)
        
        print(f"### Fazendo inferência para a amostra {i} ###")
        f1, kappa, elapsed_time = predict(model, tokenizer, df_teste)
        resultados.append({'amostra': i, 'f1_score': f1, 'kappa': kappa, 'tempo': elapsed_time})

    # Armazenar os resultados em um DataFrame
    resultados_df = pd.DataFrame(resultados)
    path_avaliacao = os.path.join("data", 'dados_avaliacao', 'dados_avaliacao_QWEN_20Dez.parquet')
    resultados_df.to_parquet(path_avaliacao, index=False)
    print("Fim do Processamento")

if __name__ == "__main__":
    main()