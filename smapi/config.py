import os

class Config:
   YANDEX_API = os.environ.get("YANDEX_API_TOKEN")
   WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR")
