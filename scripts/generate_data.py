"""Генерирует покупателей и их оценки для каталога: data/users.csv, data/ratings.csv.
Запуск: python scripts/generate_data.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recsys.data import load_catalog          # noqa: E402
from recsys.simulate import generate          # noqa: E402

catalog = load_catalog(ROOT / "data" / "catalog_smartphones.csv")
users, ratings = generate(catalog, n_users=800, seed=42)
users.to_csv(ROOT / "data" / "users.csv", index=False)
ratings.to_csv(ROOT / "data" / "ratings.csv", index=False)
density = len(ratings) / (len(users) * len(catalog))
print(f"Покупателей: {len(users)}, оценок: {len(ratings)}, заполненность матрицы: {density:.1%}")
print(ratings["rating"].value_counts().sort_index().to_string())
print(users["segment"].value_counts().to_string())
