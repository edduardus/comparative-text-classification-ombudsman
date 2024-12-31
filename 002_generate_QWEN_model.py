import os
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset, DatasetDict
from utils.config import label2id, id2label
import time
import logging
import transformers
import datasets
import sys

# Ignorar warnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configurar variáveis de ambiente para desativar warnings específicos
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Configurações do modelo
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_PATH_OUTPUT_MODEL = os.path.join('modelos', 'modelos_QWEN') 

def load_data(fold):
    path_treino = os.path.join("data", "data_cross_validation", f"df_treino_fold_{fold}.parquet")
    path_validacao = os.path.join("data", "data_cross_validation", f"df_val_fold_{fold}.parquet")
    df_treino = pd.read_parquet(path_treino)
    df_validacao = pd.read_parquet(path_validacao)
    df_treino['text'] = df_treino['TxtFatoManifestacao']
    df_validacao['text'] = df_validacao['TxtFatoManifestacao']
    df_treino['label'] = df_treino['DescTipoManifestacao'].map(label2id)
    df_validacao['label'] = df_validacao['DescTipoManifestacao'].map(label2id)
    df_treino = df_treino[['text', 'label']]
    df_validacao = df_validacao[['text', 'label']]
    return df_treino, df_validacao

def preprocess_data(df_treino, df_validacao):
    dataset_train = Dataset.from_pandas(df_treino)
    dataset_test = Dataset.from_pandas(df_validacao)
    dataset_dict = DatasetDict({
        "train": dataset_train,
        "test": dataset_test
    })
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenized_datasets = dataset_dict.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_datasets, data_collator, tokenizer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"f1": f1}

def configure_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id, use_cache=False, torch_dtype=torch.bfloat16
    )
    #lora_config = LoraConfig(
    #    task_type=TaskType.SEQ_CLS,
    #    r=16,
    #    lora_alpha=32,
    #    lora_dropout=0.1,
    #)
    #model = get_peft_model(model, lora_config)
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.gradient_checkpointing_enable()
    return model

def train_model(df_treino, df_validacao, fold):
    modelo_saida = os.path.join(BASE_PATH_OUTPUT_MODEL, f"qwen_model_fold_{fold}")
    tokenized_datasets, data_collator, tokenizer = preprocess_data(df_treino, df_validacao)
    model = configure_model()
    training_args = TrainingArguments(
        output_dir=modelo_saida,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        #max_grad_norm=1.0,
        num_train_epochs=2,
        weight_decay=0.001,
        eval_strategy="steps",  
        save_strategy="steps",
        lr_scheduler_type = "linear",
        logging_steps=10,
        warmup_ratio=0.03,
        optim = "adamw_torch",
        report_to = "wandb",
        save_steps = 250,
        eval_steps = 250,
        fp16=False,
        bf16=True,
        tf32 = False,
        load_best_model_at_end=True,
        
        metric_for_best_model="eval_f1",  # Especificar a métrica para determinar o melhor modelo
        greater_is_better=True,  # Indicar que valores maiores da métrica são melhores
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    bestmodel = os.path.join(modelo_saida, "best_model")
    trainer.save_model(bestmodel)
    
    # Avaliar o melhor modelo
    metrics = trainer.evaluate()
    best_f1 = metrics["eval_f1"]
    return best_f1

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
 
    transformers.utils.logging.set_verbosity_info()
    #logging.setLevel(logging.DEBUG)
    datasets.utils.logging.set_verbosity(logging.DEBUG)
    transformers.utils.logging.set_verbosity(logging.DEBUG)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    print("Início do Processamento.")
    
    path_tempos = os.path.join("data", 'training_data', 'training_data_qwen.parquet')
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
        df_treino, df_validacao = load_data(fold+1)
        f1 = train_model(df_treino, df_validacao, fold+1)
        
        end_time = time.time()  # Tempo de fim
        elapsed_time = end_time - start_time  # Tempo decorrido
        tempos.append(elapsed_time)
        f1_scores.append(f1)
        #print(f"Tempo para o fold {fold+1}: {elapsed_time:.2f} segundos")
        #print(f"F1 Score para o fold {fold+1}: {f1:.4f}")
        
        # Salvar os tempos em um arquivo parquet a cada iteração
        tempos_df = pd.DataFrame({'fold': range(1, len(tempos) + 1), 'training_time': tempos, 'f1_score_val': f1_scores})
        tempos_df.to_parquet(path_tempos, index=False)
        #print(f"Tempos salvos em {path_tempos}")
    print("Processamento concluído.")
    

if __name__ == "__main__":
    main()