"""Подбор для нового покупателя (холодный старт): потребности и бюджет -> TOPSIS,
затем уточнение по опыту похожих покупателей (кластер K-means)."""
import numpy as np
import pandas as pd

from .clustering import behavior_features
from .hybrid import to_unit
from .mcdm import rank_by_needs


def match_segment(catalog, liked_idx, X_raw, mu, sd, centers):
    """Ставит нового покупателя в ближайший кластер.
    Профиль строится так же, как у старых покупателей, но по товарам, которые подобрал TOPSIS
    (как будто покупатель их выбрал и оценил на 5). Средняя оценка и число оценок
    берутся средними по всем покупателям, чтобы не влиять на выбор кластера."""
    pseudo = pd.DataFrame({"user_idx": 0, "item_idx": list(liked_idx), "rating": 5})
    x, _ = behavior_features(pseudo, catalog, 1)
    x[0, 1:3] = X_raw[:, 1:3].mean(axis=0)
    z = (x - mu) / sd
    d = ((centers - z) ** 2).sum(1)
    return int(d.argmin()), d


def advise(catalog, criteria, weights, allowed, segment_scores=None, beta=0.3, n=10):
    """Итог = (1 - β) · TOPSIS + β · популярность в сегменте (обе части 0..1 среди допустимых товаров).
    segment_scores - вектор по всем товарам каталога или None (если нет истории покупок)."""
    table = rank_by_needs(catalog, criteria, weights, allowed)
    if len(table) == 0:
        return table
    t = table["TOPSIS"].to_numpy()
    table["TOPSIS (0..1)"] = to_unit(t[None])[0] if len(t) > 1 else np.ones(1)
    if segment_scores is not None and len(t) > 1:
        seg = np.asarray(segment_scores)[table["index"].to_numpy()]
        table["Сегмент (0..1)"] = to_unit(seg[None])[0]
        table["Итог"] = (1 - beta) * table["TOPSIS (0..1)"] + beta * table["Сегмент (0..1)"]
    else:
        table["Итог"] = table["TOPSIS (0..1)"]
    table = table.sort_values(["Итог", "TOPSIS"], ascending=False).reset_index(drop=True)
    table.insert(0, "Место", np.arange(1, len(table) + 1))
    return table.head(n) if n else table
