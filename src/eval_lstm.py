%%writefile /content/text-autocomplete/src/eval_lstm.py
import torch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import pandas as pd

def safe_generate(model, prompt_tokens, max_length, device):
    """
    Безопасная генерация с проверками
    """
    if prompt_tokens is None or prompt_tokens.size(1) == 0:
        return None
    
    try:
        model.eval()
        with torch.no_grad():
            # Проверяем размерность
            if prompt_tokens.dim() == 1:
                prompt_tokens = prompt_tokens.unsqueeze(0)
            
            # Генерируем
            generated = model.generate(
                prompt_tokens, 
                max_length=max_length,
                device=device
            )
            
            # Проверяем результат
            if generated is not None and generated.size(1) > prompt_tokens.size(1):
                return generated
            else:
                return None
                
    except Exception as e:
        # print(f"Ошибка генерации: {e}")
        return None

def generate_completion_safe(model, tokenizer, text, split_ratio=0.75, max_new_tokens=20, device='cpu'):
    """
    Безопасная версия с множеством проверок
    """
    # 1. Проверка текста
    if not isinstance(text, str):
        return None, None
    
    text = text.strip()
    if len(text) < 20:  # Слишком короткий текст
        return None, None
    
    try:
        # 2. Токенизация
        tokens = tokenizer.encode(text, return_tensors='pt').to(device)
        
        # 3. Проверка токенов
        if tokens.size(1) < 10:  # Минимум 10 токенов
            return None, None
        
        # 4. Вычисление точки разделения
        split_point = int(tokens.size(1) * split_ratio)
        split_point = max(3, split_point)  # Минимум 3 токена для промпта
        split_point = min(split_point, tokens.size(1) - 3)  # Оставляем минимум 3 токена для таргета
        
        prompt_tokens = tokens[:, :split_point]
        target_tokens = tokens[:, split_point:]
        
        # 5. Финальные проверки
        if prompt_tokens.size(1) < 3 or target_tokens.size(1) < 3:
            return None, None
        
        # 6. Генерация
        generated = safe_generate(model, prompt_tokens, max_new_tokens, device)
        
        if generated is None:
            return None, None
        
        # 7. Декодирование
        prompt_text = tokenizer.decode(prompt_tokens[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        target_text = tokenizer.decode(target_tokens[0], skip_special_tokens=True)
        
        # 8. Извлечение продолжения
        if len(generated_text) > len(prompt_text):
            continuation = generated_text[len(prompt_text):].strip()
        else:
            continuation = generated_text.strip()
        
        if len(continuation) < 3:  # Слишком короткое продолжение
            return None, None
            
        return continuation, target_text
        
    except Exception as e:
        # print(f"Ошибка в generate_completion_safe: {e}")
        return None, None

def calculate_rouge_lstm(model, tokenizer, test_texts, split_ratio=0.75, max_new_tokens=20, device='cpu', num_samples=100):
    """
    Расчет ROUGE-1 и ROUGE-2 для LSTM модели
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': []}
    examples = []
    
    model.eval()
    
    # Берем ограниченное количество примеров
    texts_to_eval = test_texts[:num_samples]
    
    # Предварительная фильтрация
    valid_texts = []
    for t in texts_to_eval:
        if isinstance(t, str) and len(t.strip()) > 50:  # Увеличили порог
            valid_texts.append(t)
    
    print(f"Всего текстов: {len(texts_to_eval)}")
    print(f"После фильтрации по длине (>50 символов): {len(valid_texts)}")
    
    if len(valid_texts) == 0:
        print("Нет валидных текстов для оценки!")
        return {'rouge1': 0.0, 'rouge2': 0.0}, []
    
    success_count = 0
    error_count = 0
    
    for i, text in enumerate(tqdm(valid_texts, desc="Оценка LSTM")):
        try:
            generated, target = generate_completion_safe(
                model, tokenizer, text, split_ratio, max_new_tokens, device
            )
            
            if generated and target and len(generated) > 3 and len(target) > 3:
                rouge_scores = scorer.score(target, generated)
                scores['rouge1'].append(rouge_scores['rouge1'].fmeasure)
                scores['rouge2'].append(rouge_scores['rouge2'].fmeasure)
                success_count += 1
                
                # Сохраняем примеры
                if len(examples) < 5 and len(generated) > 5:
                    words = text.split()
                    split_idx = min(int(len(words) * split_ratio), len(words) - 1)
                    prompt = ' '.join(words[:split_idx])
                    examples.append({
                        'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        'generated': generated[:100] + "..." if len(generated) > 100 else generated,
                        'target': target[:100] + "..." if len(target) > 100 else target
                    })
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            if error_count < 5:  # Печатаем первые ошибки
                print(f"\nОшибка в примере {i}: {type(e).__name__}")
            continue
    
    print(f"\nУспешно: {success_count}, Ошибок: {error_count}")
    
    if success_count == 0:
        print("ВНИМАНИЕ: Не удалось обработать ни одного примера!")
        return {'rouge1': 0.0, 'rouge2': 0.0}, []
    
    avg_scores = {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2'])
    }
    
    return avg_scores, examples