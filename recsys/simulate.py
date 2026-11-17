"""Генератор покупателей и их оценок.

Реальные оценки покупателей маркетплейсов не публикуются вместе с идентификаторами людей,
поэтому история покупок моделируется. Модель прозрачная:
у каждого покупателя есть скрытый сегмент, бюджет, личные веса критериев и любимые бренды.
Полезность товара для покупателя = взвешенная сумма нормированных характеристик
+ симпатия к бренду - штраф за выход из бюджета. Покупатель чаще оценивает то, что ему
подходит и что популярно, а оценка 1..5 растёт с полезностью.
Скрытый сегмент хранится только для проверки кластеризации и не используется методами.
"""
import numpy as np
import pandas as pd

from .data import minmax_benefit

FEATURES = ["antutu", "camera", "battery", "ram", "storage", "refresh", "screen", "weight", "nfc", "year"]

# Сегменты: доля, медианный бюджет, веса критериев (без цены), любимые бренды
SEGMENTS = {
    "Экономный": dict(share=0.26, budget=13000, w=dict(battery=4, storage=2, ram=2, screen=2, nfc=2),
                      brands={"Xiaomi": .4, "POCO": .3, "realme": .3, "Tecno": .3, "Infinix": .3}),
    "Геймер": dict(share=0.18, budget=38000, w=dict(antutu=6, refresh=3, ram=3, battery=3),
                   brands={"POCO": .5, "Xiaomi": .3, "OnePlus": .4, "realme": .3}),
    "Фотограф": dict(share=0.16, budget=60000, w=dict(camera=5, antutu=2, storage=3, screen=1, year=2),
                     brands={"Google": .6, "Honor": .4, "Huawei": .4, "Samsung": .3}),
    "Премиум": dict(share=0.16, budget=105000, w=dict(antutu=3, year=4, weight=3, camera=2),
                    brands={"Apple": 1.0, "Samsung": .5}),
    "Практичный": dict(share=0.24, budget=27000, w=dict(battery=4, nfc=3, storage=3, ram=1, year=1),
                       brands={"Samsung": .4, "Honor": .3, "Huawei": .2, "Motorola": .2}),
}


BRAND_POPULARITY = {"Samsung": 1.0, "Xiaomi": 1.0, "Apple": 0.9, "POCO": 0.7, "Honor": 0.7, "realme": 0.6,
                    "Tecno": 0.5, "Infinix": 0.4, "Huawei": 0.4, "vivo": 0.4, "OPPO": 0.4, "OnePlus": 0.3,
                    "Motorola": 0.3, "Google": 0.25, "Nothing": 0.25}


def _utility_matrix(catalog, users_w, budgets, brand_aff):
    F = minmax_benefit(catalog, FEATURES)                       # items x features, 1 = лучшее
    U = users_w @ F.T                                           # users x items, 0..1
    price = catalog["price"].to_numpy(dtype=float)
    ratio = price[None, :] / budgets[:, None]
    over = np.maximum(0, ratio - 1.0)
    U = U - 2.5 * over - 0.25 * np.maximum(0, 0.35 - ratio)     # дорого: сильный штраф; слишком дёшево: слабый
    brands = catalog["brand"].to_numpy()
    for b, col in brand_aff.items():
        U = U + np.outer(col, (brands == b).astype(float))
    return U


def generate(catalog: pd.DataFrame, n_users=800, seed=42, mean_ratings=12):
    """Возвращает (users, ratings). users: user_id, segment, budget; ratings: user_id, item_id, rating."""
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s]["share"] for s in names])
    seg = rng.choice(len(names), size=n_users, p=shares / shares.sum())

    W = np.zeros((n_users, len(FEATURES)))
    budgets = np.zeros(n_users)
    all_brands = sorted(catalog["brand"].unique())
    brand_aff = {b: np.zeros(n_users) for b in all_brands}
    for u in range(n_users):
        s = SEGMENTS[names[seg[u]]]
        base = np.array([s["w"].get(f, 0.3) for f in FEATURES], dtype=float)
        W[u] = rng.dirichlet(base * 6)                           # личные веса около весов сегмента
        budgets[u] = s["budget"] * np.exp(rng.normal(0, 0.3))
        for b, a in s["brands"].items():
            if b in brand_aff:
                brand_aff[b][u] = a * rng.uniform(0.3, 1.0)
        for b in all_brands:                                     # случайные личные симпатии
            brand_aff[b][u] += rng.normal(0, 0.05)

    U = _utility_matrix(catalog, W, budgets, brand_aff)
    # узнаваемость бренда: популярные бренды чаще попадаются на глаза
    popularity = catalog["brand"].map(BRAND_POPULARITY).fillna(0.3).to_numpy(dtype=float)
    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.poisson(mean_ratings) + 3, 3, 35))
        p = np.exp((U[u] - U[u].max()) / 0.2) * popularity
        p = p / p.sum()
        items = rng.choice(len(catalog), size=min(n, len(catalog)), replace=False, p=p)
        z = (U[u] - U[u].mean()) / (U[u].std() + 1e-9)
        for i in items:
            r = int(np.clip(np.rint(2.6 + 1.6 * z[i] + rng.normal(0, 0.8)), 1, 5))
            rows.append((u + 1, int(catalog["item_id"].iloc[i]), r))
    users = pd.DataFrame({"user_id": np.arange(1, n_users + 1),
                          "segment": [names[k] for k in seg],
                          "budget": np.round(budgets, -2).astype(int)})
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    return users, ratings
