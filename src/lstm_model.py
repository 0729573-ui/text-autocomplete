import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, hidden=None):
        embeddings = self.embedding(input_ids)
        lstm_out, hidden = self.lstm(embeddings, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, hidden
    
    def generate(self, start_tokens, max_length=20, temperature=1.0, device='cpu'):
        """
        Генерация текста (автодополнение)
        """
        # Проверка на пустой вход
        if start_tokens is None or start_tokens.size(1) == 0:
            print("Ошибка: пустой вход для генерации")
            return start_tokens
        
        self.eval()
        generated = start_tokens.clone().detach()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Берем последние токены (не больше 50 для контекста)
                if generated.size(1) > 50:
                    current_input = generated[:, -50:]
                else:
                    current_input = generated
                
                # Проверка что вход не пустой
                if current_input.size(1) == 0:
                    break
                
                # Прямой проход
                logits, hidden = self.forward(current_input, hidden)
                
                # Берем предсказание для последнего токена
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax для получения вероятностей
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Сэмплируем следующий токен
                try:
                    next_token = torch.multinomial(probs, num_samples=1)
                except:
                    # Если ошибка сэмплирования, берем токен с макс вероятностью
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Добавляем к последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Ограничиваем длину, чтобы избежать слишком длинных последовательностей
                if generated.size(1) > 100:
                    break
        
        return generated
