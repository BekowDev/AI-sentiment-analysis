import os
from huggingface_hub import snapshot_download

MY_TOKEN = "токен"

print("--- 🔐 АВТОРИЗАЦИЯ... ---")

if not os.path.exists("local_models"):
    os.makedirs("local_models")

models = {
    "en": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "kk": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
}

for lang, repo_id in models.items():
    print(f"\n⏳ Скачиваю {lang.upper()}: {repo_id}...")
    folder_name = repo_id.split("/")[-1]

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=f"local_models/{folder_name}",
            token=MY_TOKEN,  # Используем твой личный ключ
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"]
        )
        print(f"✅ {lang.upper()} успешно скачана!")
    except Exception as e:
        print(f"❌ ОШИБКА с {lang}: {e}")

print("\n🏁 --- ЗАГРУЗКА ЗАВЕРШЕНА ---")

# Новая модель для русского (более надежная)
REPO_ID = "blanchefort/rubert-base-cased-sentiment"
LOCAL_DIR = "local_models/rubert-base-cased-sentiment"

print(f"--- ⏳ Скачиваю замену для RU: {REPO_ID} ---")

try:
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"]
    )
    print("✅ RU модель успешно скачана!")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")
