"""Контентная фильтрация: товары описываются вектором характеристик, профиль покупателя - по его оценкам."""
import numpy as np
import pandas as pd

from .data import criteria_in, minmax_benefit

BRAND_WEIGHT = 0.6     # вклад бренда в описание товара
PRICE_WEIGHT = 1.5     # цена важна при сходстве: телефон за 10 и за 100 тыс. - разные товары


def item_features(catalog: pd.DataFrame):
    """Матрица описаний товаров: нормированные характеристики (0..1, 1 = лучше),
    ценовой уровень (лог-цена), бренд (one-hot). Центрируется и нормируется по строкам."""
    cols = [c for c in criteria_in(catalog) if c != "price"]
    F = minmax_benefit(catalog, cols)
    logp = np.log(catalog["price"].to_numpy(dtype=float))
    price_level = (logp - logp.min()) / (logp.max() - logp.min() + 1e-9)
    brands = pd.get_dummies(catalog["brand"]).to_numpy(dtype=float) * BRAND_WEIGHT
    X = np.hstack([F, PRICE_WEIGHT * price_level[:, None], brands])
    X = X - X.mean(axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def item_similarity(X):
    return X @ X.T


def user_profiles(R, X):
    """Профиль = Σ (r_uj - r̄_u) x_j: понравившееся (выше своей средней) притягивает, не понравившееся отталкивает."""
    mask = R > 0
    cnt = mask.sum(1)
    mean = np.divide(R.sum(1), cnt, out=np.zeros(len(R)), where=cnt > 0)
    W = (R - mean[:, None]) * mask
    # если пользователь всё оценил одинаково, берём просто среднее оценённых товаров
    flat = (np.abs(W).sum(1) == 0) & (cnt > 0)
    W[flat] = mask[flat].astype(float)
    return W @ X


def scores(R, X):
    """Косинусная близость профиля покупателя к каждому товару (users x items), от -1 до 1."""
    P = user_profiles(R, X)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    return P @ X.T
