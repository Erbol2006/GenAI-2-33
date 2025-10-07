# Установка необходимых библиотек
!pip install transformers accelerate torch unsloth

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# конфигурация модели
MODEL_NAME = "unsloth/DeepSeek-R1-0528-Qwen3-8B"
def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    print("Загрузка модели DeepSeek-R1 (это может занять несколько минут)...")

    #для Qwen-архитектуры используем fast tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    
    #устанавливаем pad_token если его нет
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    #загружаем модель с trust_remote_code=True
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,  
        trust_remote_code=True,     
        low_cpu_mem_usage=True
    )

    print("Модель DeepSeek-R1 успешно загружена!")
    return tok, mdl

#ЗАГРУЖАЕМ МОДЕЛЬ ОДИН РАЗ
tokenizer, model = load_model_and_tokenizer()
print("\n" + "="*60)
print("МОДЕЛЬ DEEPSEEK-R1 ГОТОВА К РАБОТЕ!")
print("Можно переходить к следующим блокам")
print("="*60)


def parse_request(request: str) -> tuple[int, str]:
    """
    Достаёт количество советов и тему из русскоязычного запроса.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Запрос должен быть непустой строкой.")
    text = request.lower().strip()
    # число советов
    m = re.search(r"\d+", text)
    num = int(m.group()) if m else 3
    # тема
    topic = ""
    for kw in (" по ", " о ", " про "):
        if kw in text:
            topic = text.split(kw, 1)[1]
            # убираем часть про нумерацию
            topic = re.split(r'пронумеруй|нумеруй|важность', topic)[0]
            break
    
    topic = topic.strip().strip('?.!,"«»')
    return num, topic

def validate_numbering(text: str, expected_count: int) -> bool:
    """
    Проверяет корректность нумерации в сгенерированном тексте.
    """
    lines = text.splitlines()
    numbered_lines = []

    # ищем все пронумерованные строки
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+[\.\)]', line):
            numbered_lines.append(line)

    #проверяем количество
    if len(numbered_lines) != expected_count:
        print(f"Ожидалось {expected_count} советов, найдено {len(numbered_lines)}")
        return False

    #проверяем последовательность (1, 2, 3...)
    for i, line in enumerate(numbered_lines, 1):
        if not line.startswith(f"{i}."):
            print(f"Нарушена последовательность: ожидалось {i}., получено {line.split('.')[0]}.")
            return False
    return True
print("Вспомогательные функции загружены!")

def generate_advice(request: str) -> str:
    """
    Генерирует советы на русском языке по запросу для DeepSeek-R1.
    ИСПРАВЛЕННЫЙ ФОРМАТ ПРОМПТА ДЛЯ QWEN!
    """
    num, topic = parse_request(request)

    if topic:
        user_msg = f"Дай {num} советов по теме: {topic}. Отвечай строго по-русски."
    else:
        user_msg = f"Дай {num} полезных советов. Отвечай строго по-русски."

    # ПРАВИЛЬНЫЙ ФОРМАТ ПРОМПТА ДЛЯ QWEN-АРХИТЕКТУРЫ
    prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=512,  # Увеличили для reasoning-модели
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # выделяем только нужное количество пунктов
    tips, lines = [], text.splitlines()
    for ln in lines:
        s = ln.strip()
        if any(s.startswith(f"{i}.") or s.startswith(f"{i})") for i in range(1, num + 1)):
            tips.append(s)
        if len(tips) == num:
            break
    return "\n".join(tips) if tips else text

print("Функция generate_advice() готова к использованию!")

def generate_prioritized_advice(request: str) -> str:
    """
    Генерирует советы с приоритезацией для DeepSeek-R1.
    """
    num, topic = parse_request(request)

    # Создаем промпт с явным указанием приоритезации
    if topic:
        user_msg = f"Дай {num} советов по теме: {topic}. Пронумеруй их по важности, где 1 - самый важный совет. Отвечай строго по-русски. Формат: только пронумерованный список."
    else:
        user_msg = f"Дай {num} полезных советов. Пронумеруй их по важности, где 1 - самый важный совет. Отвечай строго по-русски. Формат: только пронумерованный список."

    # ПРАВИЛЬНЫЙ ФОРМАТ ПРОМПТА ДЛЯ QWEN-АРХИТЕКТУРЫ
    prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=512,  # Увеличили для reasoning-модели
        do_sample=True,
        temperature=0.6,     # Чуть ниже температуру для более структурированного ответа
        top_p=0.85,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return text

print("Функция generate_prioritized_advice() готова!")

def test_basic_advice():
    """
    Тестирует базовую генерацию советов.
    """
    test_requests = [
        "Дай 3 совета по учебе",
        "Дай 2 совета по программированию",
        "Дай 4 совета о здоровом образе жизни"
    ]

    print("ТЕСТ БАЗОВОЙ ГЕНЕРАЦИИ СОВЕТОВ")
    print("=" * 50)

    for request in test_requests:
        print(f"\nЗапрос: {request}")
        start_time = time.time()

        advice = generate_advice(request)

        print("Результат:")
        print(advice)
        print(f"⏱Время генерации: {time.time() - start_time:.2f}с")
        print("-" * 40)

def test_prioritized_advice():
    """
    Тестирует генерацию советов с приоритезацией.
    """
    test_requests = [
        "Дай 3 совета по учебе, пронумеруй по важности: 1 - самый важный",
        "Дай 4 совета по тайм-менеджменту, пронумеруй по важности",
        "Дай 2 совета о здоровом образе жизни, пронумеруй по важности"
    ]

    print("\nТЕСТ ПРИОРИТИЗАЦИИ СОВЕТОВ")
    print("=" * 50)

    for request in test_requests:
        print(f"\n1. Промпт: '{request}'")
        print("2. Генерация...")
        start_time = time.time()

        advice = generate_prioritized_advice(request)

        print("3. Проверка нумерации...")
        num, _ = parse_request(request)
        is_valid = validate_numbering(advice, num)

        print("4. Результат:")
        print(advice)
        print(f"Нумерация корректна: {is_valid}")
        print(f"Время генерации: {time.time() - start_time:.2f}с")
        print("-" * 40)

print("Тестовые функции готовы!")
test_prioritized_advice()