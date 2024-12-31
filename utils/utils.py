#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.config import label2id, id2label



# Dataset customizado para PyTorch
class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)



def extract_features(texts, model, tokenizer, device, max_length=512):
    model.eval()
    features = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].to(torch.float32).cpu().numpy()  # CLS token
            features.append(cls_embedding)
    return np.vstack(features)

def prepara_dados(df, max_length=512, model_name = "neuralmind/bert-base-portuguese-cased"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Carregar modelo e tokenizer do BERTimbau    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    df['label'] = df['DescTipoManifestacao'].map(label2id)

    # Extrair embeddings
    X = extract_features(df['TxtFatoManifestacao'].tolist(), model, tokenizer, device, max_length=max_length)
    y = df['label'].values
        
    return X, y
