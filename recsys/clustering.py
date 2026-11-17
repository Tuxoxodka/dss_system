"""Кластеризация покупателей по покупательскому поведению методом K-means (k-means++)."""
import numpy as np
import pandas as pd

from .data import LABELS, minmax_benefit

BEHAVIOR = ["antutu", "camera", "battery", "ram", "storage", "refresh", "weight", "nfc", "year"]
PREMIUM_BRANDS = {"Apple", "Samsung", "Google"}


def behavior_features(ratings: pd.DataFrame, catalog: pd.DataFrame, n_users: int):
    """Профиль покупателя по его оценкам (без скрытого сегмента):
    средняя цена понравившегося (логарифм), средняя оценка, число оценок,
    доля премиальных брендов, средние нормированные характеристики понравившихся товаров.
    Вес товара в профиле = оценка (понравившееся сильнее влияет на профиль)."""
    cols = [c for c in BEHAVIOR if c in catalog.columns]
    F = minmax_benefit(catalog, cols)
    logp = np.log(catalog["price"].to_numpy(dtype=float))
    prem = catalog["brand"].isin(PREMIUM_BRANDS).to_numpy(dtype=float)
    names = ["Цена понравившегося (лог)", "Средняя оценка", "Число оценок (лог)", "Доля премиум-брендов"] + \
            [LABELS[c] for c in cols]
    X = np.zeros((n_users, 4 + len(cols)))
    for u, g in ratings.groupby("user_idx"):
        idx = g["item_idx"].to_numpy()
        w = g["rating"].to_numpy(dtype=float) ** 2
        w = w / w.sum()
        X[u, 0] = w @ logp[idx]
        X[u, 1] = g["rating"].mean()
        X[u, 2] = np.log(len(g))
        X[u, 3] = w @ prem[idx]
        X[u, 4:] = w @ F[idx]
    return X, names


def standardize(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def kmeans(X, k, n_init=10, max_iter=300, seed=0):
    """K-means с инициализацией k-means++. Возвращает (метки, центры, инерция)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(centers)
        for _ in range(max_iter):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            labels = d.argmin(axis=1)
            newC = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels, C, inertia)
    return best


def silhouette(X, labels):
    """Средний коэффициент силуэта: (b - a) / max(a, b)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None]) ** 2).sum(-1), 0))
    ks = np.unique(labels)
    if len(ks) < 2:
        return 0.0
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        n_same = same.sum() - 1
        if n_same == 0:
            continue
        a = D[i, same].sum() / n_same
        b = min(D[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def choose_k(X, k_range=range(2, 9), seed=0):
    """Перебор k: инерция (метод локтя) и силуэт. Лучший k - максимальный силуэт."""
    rows = []
    for k in k_range:
        labels, _, inertia = kmeans(X, k, seed=seed)
        rows.append((k, inertia, silhouette(X, labels)))
    table = pd.DataFrame(rows, columns=["k", "Инерция", "Силуэт"])
    return int(table.loc[table["Силуэт"].idxmax(), "k"]), table


def pca2(X):
    """Проекция на две главные компоненты (для графика)."""
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = (s ** 2) / (s ** 2).sum()
    return Xc @ Vt[:2].T, explained[:2]


def adjusted_rand_index(a, b):
    """ARI: согласие двух разбиений с поправкой на случайность (1 = совпадают, ~0 = случайно)."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2
    sum_ij = comb(M).sum()
    sa, sb = comb(M.sum(1)).sum(), comb(M.sum(0)).sum()
    expected = sa * sb / comb(len(a))
    max_idx = (sa + sb) / 2
    return float((sum_ij - expected) / (max_idx - expected)) if max_idx != expected else 1.0


def describe_clusters(labels, X_raw, names, ratings, catalog, users_budget_proxy=None):
    """Профиль каждого кластера: размер, типичная цена, ключевые предпочтения, любимые бренды и товары."""
    Z, _, _ = standardize(X_raw)
    rows = []
    for c in np.unique(labels):
        m = labels == c
        price = float(np.exp(X_raw[m, 0].mean()))
        z = Z[m].mean(axis=0)
        feat_idx = np.argsort(-z[4:])[:2] + 4
        likes = ", ".join(names[j].lower() for j in feat_idx if z[j] > 0.15) or "без ярких предпочтений"
        r = ratings[ratings["user_idx"].isin(np.where(m)[0])]
        top = (r.groupby("item_idx")["rating"].agg(["mean", "count"])
                .assign(score=lambda t: t["mean"] * np.log1p(t["count"]))
                .sort_values("score", ascending=False).head(3))
        brands = (catalog.loc[r.loc[r["rating"] >= 4, "item_idx"], "brand"].value_counts(normalize=True).head(3))
        rows.append({
            "Кластер": int(c), "Покупателей": int(m.sum()),
            "Типичная цена, руб.": int(round(price, -2)),
            "Средняя оценка": round(float(X_raw[m, 1].mean()), 2),
            "Важно": likes,
            "Любимые бренды": ", ".join(f"{b} {p:.0%}" for b, p in brands.items()),
            "Чаще всего хвалят": "; ".join(catalog.loc[top.index, "model"]),
        })
    return pd.DataFrame(rows)


def cluster_name(row):
    """Короткое имя кластера по типичной цене и предпочтениям."""
    p = row["Типичная цена, руб."]
    level = "бюджетный" if p < 18000 else "средний" if p < 45000 else "дорогой" if p < 80000 else "премиальный"
    return f"{level} сегмент: {row['Важно']}"
