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
    
    # Set a padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def preprocess_function(texts, tokenizer):
    return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

def predict(model, tokenizer, df):
    start_time = time.time()  # Start time
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
    # Calculate metrics
    f1 = f1_score(df["DescTipoManifestacao"].values, predicted_labels, average='weighted')
    kappa = cohen_kappa_score(df["DescTipoManifestacao"].values, predicted_labels)
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Elapsed time

    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return f1, kappa, elapsed_time

def main():
    print("Processing started")
    
    # Load the saved model for fold 1
    BASE_PATH_OUTPUT_MODEL = os.path.join('modelos', 'modelos_QWEN')
    model_name = os.path.join(BASE_PATH_OUTPUT_MODEL, "qwen_model_fold_1", "best_model")
    model, tokenizer = load_model_and_tokenizer(model_name)

    resultados = []

    # Loop for inference on all 30 different datasets
    for i in range(1, 31):
        path_data_teste = os.path.join("data", "data_cross_validation", "amostra_teste", f"amostra_bootstrap_{i}.parquet")
        df_teste = pd.read_parquet(path_data_teste)
        
        print(f"### Performing inference for sample {i} ###")
        f1, kappa, elapsed_time = predict(model, tokenizer, df_teste)
        resultados.append({'sample': i, 'f1_score': f1, 'kappa': kappa, 'time': elapsed_time})

    # Store the results in a DataFrame
    resultados_df = pd.DataFrame(resultados)
    path_avaliacao = os.path.join("data", 'evaluation_data', 'evaluation_data_QWEN.parquet')
    resultados_df.to_parquet(path_avaliacao, index=False)
    print("Processing completed")

if __name__ == "__main__":
    main()