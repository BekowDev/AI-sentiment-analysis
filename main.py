import torch
import re
from fastapi import FastAPI, HTTPException, Header, Security
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from langdetect import detect, LangDetectException
# source venv/bin/activate
# uvicorn main:app --reload
# TODO: 6. Улучшение Python-скрипта Суммаризация: Используй небольшую LLM (например, Llama-3 или Saiga), чтобы она писала краткий вывод: "В целом пользователи довольны подарком, но многие жалуются на долгую доставку". Авто-теги: Группировка комментов по темам (вопросы, жалобы, благодарности).

# --- КОНФИГУРАЦИЯ ---
API_KEY_SECRET = "python_secret_key" # Замени на свой секрет

class MultilingualSentimentAnalyzer:
    def __init__(self):
        print("INIT: Определение устройства (CPU/GPU/MPS)...")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"INIT: Использую устройство: {self.device}")

        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            top_k=None
        )

        # Список матов (только жесткие оскорбления)
        self.bad_words = {
            'ru': ['дурак', 'идиот', 'урод', 'тупой', 'блять', 'сука', 'хер', 'мудак'],
            'kk': ['ақымақ', 'есек', 'топас', 'жынды', 'щешес', 'мал'],
            'en': ['stupid', 'idiot', 'fuck', 'shit', 'bitch']
        }

        # Быстрые исправления для ИИ
        self.manual_overrides = {
            "полная ерунда": "negative",
            "ерунда": "negative",
            "чушь": "negative",
            "бред": "negative"
        }

    def detect_language(self, text):
        kz_letters = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
        if any(char in kz_letters for char in text):
            return 'kk'
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    def check_toxicity(self, text, lang):
        all_bad = self.bad_words.get(lang, self.bad_words['ru'] + self.bad_words['kk'])
        text_lower = text.lower()
        for word in all_bad:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                return True
        return False

    def analyze(self, comments: List[str]):
        # Ограничиваем длину каждого коммента (защита от перегрузки)
        safe_comments = [c[:512] for c in comments]

        # Пакетная обработка через нейросеть (Batching)
        # Это дает огромный прирост скорости на больших списках
        try:
            batch_results = self.sentiment_pipeline(safe_comments, batch_size=32)
        except Exception as e:
            print(f"Ошибка модели: {e}")
            batch_results = [None] * len(comments)

        final_results = []
        label_map = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}

        for i, text in enumerate(comments):
            text_clean = text.lower().strip()
            lang = self.detect_language(text)

            # 1. Приоритет - ручные правки
            if text_clean in self.manual_overrides:
                sentiment = self.manual_overrides[text_clean]
                score = 1.0
            # 2. Результат нейросети
            elif batch_results[i]:
                top_pred = max(batch_results[i], key=lambda x: x['score'])
                sentiment = label_map.get(top_pred['label'], top_pred['label'])
                score = round(top_pred['score'], 4)
            else:
                sentiment, score = "unknown", 0.0

            final_results.append({
                "text": text,
                "sentiment": sentiment,
                "score": score,
                "language": lang,
                "is_toxic": self.check_toxicity(text, lang)
            })

        return final_results

# --- API СЕРВЕР ---
app = FastAPI(title="AI Sentiment Service")
ai_engine = MultilingualSentimentAnalyzer()

class AnalysisRequest(BaseModel):
    comments: List[str]

@app.post("/analyze")
async def analyze_api(request: AnalysisRequest, x_api_key: str = Header(None)):
    # Проверка безопасности
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    if not request.comments:
        return []

    return ai_engine.analyze(request.comments)

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(ai_engine.device)}
