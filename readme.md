rm -rf venv

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
or
pip install transformers torch langdetect scipy fastapi uvicorn sentencepiece protobuf

uvicorn main:app --reload

X-API-Key your_super_secret_key_here
