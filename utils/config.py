import torch
import torch.nn as nn

# Definição do mapeamento de rótulos (label2id e id2label)
label2id = {
    'Acesso à Informação': 0,
    'Denúncia': 1,
    'Reclamação': 2,
    'Elogio': 3,
    'Sugestão': 4,
    'Solicitação': 5    
}
id2label = {v: k for k, v in label2id.items()}


# Definição da Arquitetura da Rede CNN patra classificação de texto
class TextCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)  # Ajuste o kernel_size para 1
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Adicionar dimensão para CNN (batch_size, channels, sequence_length)
        x = x.unsqueeze(2)  # (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x = self.conv1(x)  # (batch_size, 128, 1)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)  # (batch_size, 128)
        x = self.fc(x)  # (batch_size, num_classes)
        return x
    
    
# Definição da Arquitetura da Rede LSTM para classificação de texto
class TextLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 para bidirecional

    def forward(self, x):
        # Adicionar dimensão para LSTM (batch_size, sequence_length, input_dim)
        x = x.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim)
        h0 = torch.zeros(2 * 2, x.size(0), 128).to(x.device)  # 2 para bidirecional, 2 para num_layers
        c0 = torch.zeros(2 * 2, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Pegue a última saída da sequência
        return out