from transformers import pipeline
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

def evaluate_transformer(test_texts, split_ratio=0.75, max_new_tokens=20, num_samples=100):
    """
    Оценка предобученной модели distilgpt2
    """
    print("Загрузка модели distilgpt2...")
    generator = pipeline("text-generation", model="distilgpt2", device_map="auto")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': []}
    examples = []
    
    texts_to_eval = test_texts[:num_samples]
    
    for text in tqdm(texts_to_eval, desc="Evaluating Transformer"):
        # Разделяем текст
        words = text.split()
        if len(words) < 5:
            continue
            
        split_point = int(len(words) * split_ratio)
        prompt = ' '.join(words[:split_point])
        target = ' '.join(words[split_point:])
        
        if len(target) == 0:
            continue
        
        # Генерируем продолжение
        try:
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=50256  # eos token for GPT-2
            )
            generated_full = result[0]['generated_text']
            generated_continuation = generated_full[len(prompt):].strip()
            
            if len(generated_continuation) > 0:
                rouge_scores = scorer.score(target, generated_continuation)
                scores['rouge1'].append(rouge_scores['rouge1'].fmeasure)
                scores['rouge2'].append(rouge_scores['rouge2'].fmeasure)
                
                if len(examples) < 5:
                    examples.append({
                        'prompt': prompt,
                        'generated': generated_continuation,
                        'target': target
                    })
        except Exception as e:
            print(f"Ошибка: {e}")
            continue
    
    avg_scores = {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2'])
    }
    
    return avg_scores, examples
