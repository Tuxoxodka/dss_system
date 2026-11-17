"""Каталог товаров и оценки покупателей: чтение, проверка, нормировка признаков."""
import io

import numpy as np
import pandas as pd

# Критерии товара: ключ столбца -> (подпись, единица, направление).
# max: больше = лучше, min: меньше = лучше.
CRITERIA = {
    "price":   ("Цена", "руб.", "min"),
    "antutu":  ("Производительность", "тыс. AnTuTu", "max"),
    "camera":  ("Камера", "Мп", "max"),
    "battery": ("Аккумулятор", "мА·ч", "max"),
    "ram":     ("Оперативная память", "ГБ", "max"),
    "storage": ("Встроенная память", "ГБ", "max"),
    "refresh": ("Частота экрана", "Гц", "max"),
    "screen":  ("Диагональ", "дюйм", "max"),
    "weight":  ("Вес", "г", "min"),
    "nfc":     ("NFC", "есть/нет", "max"),
    "year":    ("Год выхода", "", "max"),
}
LABELS = {k: v[0] for k, v in CRITERIA.items()}
REQUIRED = ["model", "brand", "price"]


def read_csv_any(data: bytes) -> pd.DataFrame:
    """CSV из любого редактора: UTF-8 или Windows-1251, разделитель «,» или «;»."""
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось прочитать файл: неизвестная кодировка.")
    first = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if sep == ";":  # в русском Excel дробная часть пишется через запятую
        for c in df.columns:
            if c in CRITERIA and not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    return df


def validate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет каталог и приводит его к рабочему виду.
    Обязательны столбцы model, brand, price; остальные критерии из CRITERIA необязательны."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Нет обязательных столбцов: {missing}")
    if len(df) < 2:
        raise ValueError("В каталоге должно быть минимум два товара.")
    if df["model"].duplicated().any():
        raise ValueError("Названия моделей в столбце model должны быть разными.")
    if "item_id" not in df.columns:
        df.insert(0, "item_id", np.arange(1, len(df) + 1))
    crit = [c for c in CRITERIA if c in df.columns]
    for c in crit:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = [c for c in crit if df[c].isna().any()]
    if bad:
        raise ValueError(f"В столбцах есть пустые или нечисловые значения: {bad}")
    if (df["price"] <= 0).any():
        raise ValueError("Цена должна быть больше нуля.")
    if len(crit) < 3:
        raise ValueError("Нужна цена и минимум ещё два числовых критерия.")
    df["brand"] = df["brand"].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_catalog(path) -> pd.DataFrame:
    return validate_catalog(pd.read_csv(path))


def criteria_in(df: pd.DataFrame):
    """Критерии, которые есть в каталоге, в порядке CRITERIA."""
    return [c for c in CRITERIA if c in df.columns]


def minmax_benefit(df: pd.DataFrame, cols):
    """Нормировка 0..1 так, что 1 всегда лучшее значение (для min-критериев шкала переворачивается)."""
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        x = df[c].to_numpy(dtype=float)
        lo, hi = x.min(), x.max()
        z = np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)
        out[:, j] = 1 - z if CRITERIA[c][2] == "min" else z
    return out


def rating_matrix(ratings: pd.DataFrame, n_users: int, n_items: int):
    """Плотная матрица оценок users x items (0 = нет оценки). Индексы с нуля."""
    R = np.zeros((n_users, n_items))
    R[ratings["user_idx"].to_numpy(), ratings["item_idx"].to_numpy()] = ratings["rating"].to_numpy()
    return R


def index_ratings(ratings: pd.DataFrame, catalog: pd.DataFrame, users: pd.DataFrame):
    """Добавляет к оценкам индексы строк каталога и пользователей."""
    item_pos = {iid: k for k, iid in enumerate(catalog["item_id"])}
    user_pos = {uid: k for k, uid in enumerate(users["user_id"])}
    r = ratings[ratings["item_id"].isin(item_pos) & ratings["user_id"].isin(user_pos)].copy()
    r["item_idx"] = r["item_id"].map(item_pos)
    r["user_idx"] = r["user_id"].map(user_pos)
    return r.reset_index(drop=True)
