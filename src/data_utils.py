import pandas as pd
import re
import emoji
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
import csv

def load_raw_dataset(file_path='/content/text-autocomplete/data/raw_dataset.csv'):
    """
    Загрузка сырого датасета
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', header=None, names=['text'])
    except:
        try:
            df = pd.read_csv(file_path, encoding='latin-1', header=None, names=['text'])
        except:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
            df = pd.DataFrame({'text': texts})
    
    print(f"Загружено {len(df)} записей")
    return df

def clean_text(text):
    """
    Очистка текста
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) < 3:
        return ""
    
    return text

def tokenize_texts(texts, tokenizer_name='distilgpt2', max_length=128):
    """
    Токенизация текстов
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_texts = []
    all_tokens = []
    
    print("Токенизация текстов...")
    for text in tqdm(texts):
        if text and len(text) > 0:
            tokens = tokenizer.encode(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )
            if len(tokens) > 3:
                tokenized_texts.append({
                    'text': text,
                    'tokens': tokens,
                    'length': len(tokens)
                })
                all_tokens.extend(tokens)
    
    print(f"Токенизировано {len(tokenized_texts)} текстов")
    return tokenized_texts, tokenizer

def create_sequences(token_list, seq_length=5):
    """
    Создание последовательностей X и Y
    """
    X, y = [], []
    
    for i in range(len(token_list) - seq_length):
        X.append(token_list[i:i+seq_length])
        y.append(token_list[i+1:i+seq_length+1])
    
    return np.array(X), np.array(y)

def prepare_dataset_for_training(tokenized_texts, seq_length=5, test_size=0.2, val_size=0.1):
    """
    Подготовка датасета для обучения
    """
    all_X, all_y = [], []
    
    print("Создание обучающих последовательностей...")
    for item in tqdm(tokenized_texts):
        tokens = item['tokens']
        if len(tokens) > seq_length:
            X, y = create_sequences(tokens, seq_length)
            all_X.extend(X)
            all_y.extend(y)
    
    all_X = np.array(all_X)
    all_y = np.array(all_y)
    
    print(f"Всего примеров: {len(all_X)}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_X, all_y, test_size=test_size, random_state=42
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42
    )
    
    print(f"Train: {len(X_train)}")
    print(f"Val: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

def save_processed_data(datasets, tokenizer, output_dir='/content/text-autocomplete/data/'):
    """
    Сохранение обработанных данных
    """
    tokenizer.save_pretrained(output_dir + 'tokenizer/')
    
    for name, (X, y) in datasets.items():
        np.save(f'{output_dir}{name}_X.npy', X)
        np.save(f'{output_dir}{name}_y.npy', y)
        print(f"Сохранено {name}: {X.shape}")

def load_processed_data(data_dir='/content/text-autocomplete/data/'):
    """
    Загрузка обработанных данных
    """
    datasets = {}
    for name in ['train', 'val', 'test']:
        try:
            X = np.load(f'{data_dir}{name}_X.npy')
            y = np.load(f'{data_dir}{name}_y.npy')
            datasets[name] = (X, y)
            print(f"Загружено {name}: {X.shape}")
        except FileNotFoundError:
            print(f"Файл {name}_X.npy не найден")
            return None
    
    return datasets

def process_pipeline(raw_file='/content/text-autocomplete/data/raw_dataset.csv', 
                     processed_file='/content/text-autocomplete/data/dataset_processed.csv',
                     seq_length=5):
    """
    Полный пайплайн обработки данных
    """
    print("="*50)
    print("НАЧАЛО ОБРАБОТКИ ДАННЫХ")
    print("="*50)
    
    print("\n1. Загрузка сырых данных...")
    df_raw = load_raw_dataset(raw_file)
    
    print("\n2. Очистка текстов...")
    tqdm.pandas(desc="Очистка")
    df_raw['cleaned_text'] = df_raw['text'].progress_apply(clean_text)
    
    df_cleaned = df_raw[df_raw['cleaned_text'].str.len() > 0].copy()
    print(f"После очистки осталось {len(df_cleaned)} записей")
    
    df_cleaned[['cleaned_text']].to_csv(processed_file, index=False)
    print(f"Очищенный датасет сохранен в {processed_file}")
    
    print("\n3. Токенизация текстов...")
    tokenized_texts, tokenizer = tokenize_texts(df_cleaned['cleaned_text'].tolist())
    
    print("\n4. Подготовка обучающих примеров...")
    datasets = prepare_dataset_for_training(tokenized_texts, seq_length)
    
    print("\n5. Сохранение обработанных данных...")
    save_processed_data(datasets, tokenizer)
    
    print("\n" + "="*50)
    print("ОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА")
    print("="*50)
    
    return datasets, tokenizer
