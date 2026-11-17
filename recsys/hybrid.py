"""Гибридная модель: взвешивание + переключение.

0 оценок           -> холодный старт: подбор по потребностям (TOPSIS) + популярное в сегменте похожих покупателей;
1..4 оценки        -> CF = item-kNN по взаимодействиям (работает с одной-двумя покупками);
5 и более оценок   -> CF = среднее item-kNN и iALS (факторизация надёжна, когда истории достаточно);
итог               -> α · CF + (1 - α) · CB, все части нормированы в 0..1 по ещё не оценённым товарам.
"""
import warnings

import numpy as np

DEFAULT_ALPHA = 0.9
MIN_RATINGS_FOR_ALS = 5


def to_unit(S, R=None):
    """Нормировка оценок каждого пользователя в 0..1 по ещё не оценённым товарам."""
    S = np.asarray(S, dtype=float)
    M = S if R is None else np.where(R > 0, np.nan, S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(M, axis=1, keepdims=True)
        hi = np.nanmax(M, axis=1, keepdims=True)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return np.clip((S - lo) / span, 0, 1)


def blend(cf_pred, cb_score, R, alpha=DEFAULT_ALPHA):
    """Взвешенный гибрид α·CF + (1-α)·CB. alpha - число или массив по пользователям."""
    cf = to_unit(cf_pred, R)
    cb = to_unit(cb_score, R)
    a = np.broadcast_to(np.asarray(alpha, dtype=float), (len(R),))
    return a[:, None] * cf + (1 - a[:, None]) * cb


def cf_switching(knn_scores, als_scores, R, min_ratings=MIN_RATINGS_FOR_ALS):
    """Коллаборативная часть с переключением по длине истории покупателя."""
    knn = to_unit(knn_scores, R)
    als = to_unit(als_scores, R)
    n = (R > 0).sum(1)
    rich = (n >= min_ratings)[:, None]
    return np.where(rich, 0.5 * knn + 0.5 * als, knn)


def hybrid_scores(knn_scores, als_scores, cb_score, R, alpha=DEFAULT_ALPHA, min_ratings=MIN_RATINGS_FOR_ALS):
    return blend(cf_switching(knn_scores, als_scores, R, min_ratings), cb_score, R, alpha)


def top_n(scores_row, exclude=None, allowed=None, n=10):
    """Индексы n лучших товаров с учётом исключений (уже оценённые) и допустимых (бюджет, фильтры)."""
    s = np.asarray(scores_row, dtype=float).copy()
    if exclude is not None:
        s[np.asarray(exclude, dtype=bool)] = -np.inf
    if allowed is not None:
        s[~np.asarray(allowed, dtype=bool)] = -np.inf
    order = np.argsort(-s, kind="stable")
    return [int(i) for i in order[:n] if np.isfinite(s[i])]


def cluster_popularity(ratings, labels, n_items, prior=3.0):
    """Байесовское среднее оценок товара внутри каждого кластера × log(1 + число оценок):
    товар, который в сегменте часто покупают и хвалят, получает высокий балл."""
    k = int(labels.max()) + 1
    out = np.zeros((k, n_items))
    for c in range(k):
        r = ratings[ratings["user_idx"].isin(np.where(labels == c)[0])]
        C = r["rating"].mean() if len(r) else 3.0
        g = r.groupby("item_idx")["rating"].agg(["sum", "count"])
        mean = np.full(n_items, C)
        cnt = np.zeros(n_items)
        mean[g.index.to_numpy()] = (g["sum"] + prior * C) / (g["count"] + prior)
        cnt[g.index.to_numpy()] = g["count"]
        out[c] = mean * np.log1p(cnt)
    return out


def explain_cf(u, i, R, item_sim, catalog, n=1):
    """Объяснение: какой из понравившихся пользователю товаров больше всего похож на рекомендованный."""
    rated = np.where(R[u] >= 4)[0]
    if len(rated) == 0:
        rated = np.where(R[u] > 0)[0]
    if len(rated) == 0:
        return ""
    best = rated[np.argsort(-item_sim[i, rated])[:n]]
    return "; ".join(f"его часто выбирают те, кто купил {catalog['model'].iloc[j]} (ваша оценка {int(R[u, j])})"
                     for j in best)
