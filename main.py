import torch
import re
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline
# from langdetect import detect, LangDetectException # Можно раскомментировать, если библиотека установлена

# --- КОНФИГУРАЦИЯ ---
API_KEY_SECRET = "python_secret_key"

class MultilingualSentimentAnalyzer:
    def __init__(self):
        print("INIT: Определение устройства (CPU/GPU/MPS)...")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ INIT: Использую Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✅ INIT: Использую NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("⚠️ INIT: Использую CPU (Медленно)")

        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        # Загружаем пайплайн
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            top_k=None # Возвращаем все оценки, чтобы найти max
        )

        # Список матов
        self.bad_words = {
            'ru': ['дурак', 'идиот', 'урод', 'тупой', 'блять', 'сука', 'хер', 'мудак'],
            'kk': ['ақымақ', 'есек', 'топас', 'жынды', 'щешес', 'мал'],
            'en': ['stupid', 'idiot', 'fuck', 'shit', 'bitch']
        }

        # Ручные правила
        self.manual_overrides = {
            "полная ерунда": "negative",
            "ерунда": "negative",
            "чушь": "negative",
            "бред": "negative"
        }

    def detect_language(self, text):
        # Упрощенная детекция для скорости (можно вернуть langdetect)
        kz_letters = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
        if any(char in kz_letters for char in text):
            return 'kk'
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return 'ru' # По умолчанию считаем RU, если нет специфичных признаков

    def check_toxicity(self, text, lang):
        all_bad = self.bad_words.get(lang, self.bad_words['ru'] + self.bad_words['kk'])
        text_lower = text.lower()
        for word in all_bad:
            # Ищем слово целиком, чтобы не банить "оскорблять" из-за "бля"
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                return True
        return False

    def analyze(self, comments: List[str]):
        results = [None] * len(comments)
        indices_to_process = []
        texts_to_process = []

        # 🔥 ИСПРАВЛЕНИЕ 1: Расширенный маппинг меток
        # Модель может возвращать разные форматы, учтем их все
        label_map = {
            'LABEL_0': 'negative', '0': 'negative', 'negative': 'negative', 'Negative': 'negative',
            'LABEL_1': 'neutral',  '1': 'neutral',  'neutral': 'neutral',  'Neutral': 'neutral',
            'LABEL_2': 'positive', '2': 'positive', 'positive': 'positive', 'Positive': 'positive'
        }

        # 1. Предварительный проход (ручные правила)
        for i, text in enumerate(comments):
            text_clean = text.lower().strip()
            lang = self.detect_language(text)

            # Проверяем на пустоту
            if not text_clean:
                results[i] = {"text": text, "sentiment": "neutral", "score": 0.0, "language": lang, "is_toxic": False}
                continue

            if text_clean in self.manual_overrides:
                results[i] = {
                    "text": text,
                    "sentiment": self.manual_overrides[text_clean],
                    "score": 1.0,
                    "language": lang,
                    "is_toxic": self.check_toxicity(text, lang)
                }
            else:
                indices_to_process.append(i)
                texts_to_process.append(text[:512])

        # 2. Нейросеть
        if texts_to_process:
            batch_size = 16
            with torch.no_grad():
                for j in range(0, len(texts_to_process), batch_size):
                    batch_texts = texts_to_process[j : j + batch_size]
                    batch_indices = indices_to_process[j : j + batch_size]

                    try:
                        predictions = self.sentiment_pipeline(batch_texts)

                        for k, pred_list in enumerate(predictions):
                            # Если pipeline возвращает список списков (иногда бывает), берем топ
                            if isinstance(pred_list, list):
                                top_pred = max(pred_list, key=lambda x: x['score'])
                            else:
                                top_pred = pred_list # Иногда возвращает сразу dict

                            raw_label = top_pred['label']

                            # 🔥 ИСПРАВЛЕНИЕ 2: Лог в консоль (увидите это в терминале VS Code)
                            if j == 0 and k == 0:
                                print(f"🔍 DEBUG: Модель вернула метку: '{raw_label}'")

                            sentiment = label_map.get(raw_label, 'neutral') # Если не нашли, будет neutral

                            original_idx = batch_indices[k]
                            original_text = comments[original_idx]
                            lang = self.detect_language(original_text)

                            results[original_idx] = {
                                "text": original_text,
                                "sentiment": sentiment,
                                "score": round(top_pred['score'], 4),
                                "language": lang,
                                "is_toxic": self.check_toxicity(original_text, lang)
                            }

                        if self.device.type == 'mps':
                            torch.mps.empty_cache()

                    except Exception as e:
                        print(f"❌ Ошибка в батче: {e}")
                        # Чтобы фронт не падал, заполняем ошибками
                        for idx in batch_indices:
                             results[idx] = {
                                "text": comments[idx],
                                "sentiment": "neutral", # Фолбэк на нейтральный при ошибке
                                "score": 0.0,
                                "language": "unknown",
                                "is_toxic": False
                            }

        return results
# --- API СЕРВЕР ---
app = FastAPI(title="AI Sentiment Service")
ai_engine = MultilingualSentimentAnalyzer()

class AnalysisRequest(BaseModel):
    comments: List[str]

@app.post("/analyze")
async def analyze_api(request: AnalysisRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    if not request.comments:
        return []

    return ai_engine.analyze(request.comments)

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(ai_engine.device)}
