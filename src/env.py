import os
from pathlib import Path

cache_path = Path("C:\\tempjeka\\cache")

os.environ["TORCH_HOME"] = str(cache_path / "models")
os.environ["HF_HOME"] = str(cache_path / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(cache_path / "huggingface")