"""Подбор товара под потребности и бюджет: ограничения, AHP (Саати), TOPSIS, Парето."""
import numpy as np
import pandas as pd

from .data import CRITERIA

# Показатель случайной согласованности (ПСС) из методички, n = размер матрицы
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54}


def apply_constraints(catalog, budget=(0, np.inf), need_nfc=False, min_ram=0, min_storage=0,
                      exclude_brands=()):
    """Допустимые альтернативы: в бюджете и с обязательными характеристиками."""
    ok = catalog["price"].between(*budget)
    if need_nfc and "nfc" in catalog:
        ok &= catalog["nfc"] == 1
    if "ram" in catalog:
        ok &= catalog["ram"] >= min_ram
    if "storage" in catalog:
        ok &= catalog["storage"] >= min_storage
    if exclude_brands:
        ok &= ~catalog["brand"].isin(exclude_brands)
    return ok.to_numpy()


# ---------- AHP ----------

def pairwise_from_points(points):
    """Матрица парных сравнений по баллам важности 1..5:
    разница баллов 0 -> 1, 1 -> 3, 2 -> 5, 3 -> 7, 4 -> 9; ниже диагонали обратные значения."""
    p = np.asarray(points, dtype=float)
    d = p[:, None] - p[None, :]
    v = np.minimum(1 + 2 * np.abs(d), 9)
    return np.where(d >= 0, v, 1.0 / v)


def matrix_from_upper(n, upper):
    M = np.ones((n, n))
    for (i, j), v in upper.items():
        M[i, j] = v
        M[j, i] = 1.0 / v
    return M


def ahp_weights(M):
    """Веса по методичке: среднее геометрическое строк / сумма;
    λmax = Σ (сумма столбца × вес); CI = (λmax - n)/(n - 1); CR = CI / ПСС."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    geo = np.prod(M, axis=1) ** (1.0 / n)
    w = geo / geo.sum()
    lam = float((M.sum(axis=0) * w).sum())
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.54)
    return w, lam, ci, (ci / ri if ri > 0 else 0.0)


# ---------- TOPSIS ----------

def topsis(X, directions, w):
    """C = D⁻ / (D⁺ + D⁻): близость к идеальному решению. Возвращает C, D⁺, D⁻."""
    X = np.asarray(X, dtype=float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    V = X / norm * np.asarray(w, dtype=float)
    is_max = np.array([d == "max" for d in directions])
    best = np.where(is_max, V.max(axis=0), V.min(axis=0))
    worst = np.where(is_max, V.min(axis=0), V.max(axis=0))
    d_plus = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - worst) ** 2).sum(axis=1))
    den = d_plus + d_minus
    c = np.divide(d_minus, den, out=np.full_like(den, 0.5), where=den > 0)
    return c, d_plus, d_minus


def contributions(X, directions, w):
    """Вклад критериев в оценку товара (для объяснения): w_j × нормированное значение 0..1."""
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Z = (X - lo) / span
    for j, d in enumerate(directions):
        if d == "min":
            Z[:, j] = 1 - Z[:, j]
    return Z * np.asarray(w, dtype=float)


# ---------- Парето ----------

def pareto_front(X, directions):
    """True для недоминируемых альтернатив."""
    S = np.asarray(X, dtype=float).copy()
    for j, d in enumerate(directions):
        if d == "min":
            S[:, j] = -S[:, j]
    m = len(S)
    res = np.ones(m, dtype=bool)
    for a in range(m):
        dominated = ((S >= S[a]).all(axis=1) & (S > S[a]).any(axis=1))
        res[a] = not dominated.any()
    return res


def rank_by_needs(catalog, criteria, weights, allowed):
    """TOPSIS и Парето среди допустимых товаров. Возвращает таблицу (по убыванию оценки)."""
    sub = catalog[allowed].reset_index()
    if len(sub) == 0:
        return sub
    X = sub[criteria].to_numpy(dtype=float)
    dirs = [CRITERIA[c][2] for c in criteria]
    c, dp, dm = topsis(X, dirs, weights)
    contrib = contributions(X, dirs, weights)
    sub["TOPSIS"] = c
    sub["Парето"] = pareto_front(X, dirs)
    labels = [CRITERIA[k][0] for k in criteria]
    sub["Сильные стороны"] = [", ".join(labels[j] for j in np.argsort(-row)[:3]) for row in contrib]
    return sub.sort_values("TOPSIS", ascending=False).reset_index(drop=True)
