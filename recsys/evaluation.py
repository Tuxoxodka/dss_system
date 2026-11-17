"""Оценка качества рекомендаций: точность прогноза оценок и метрики ранжирования."""
import numpy as np
import pandas as pd

from . import content
from .collaborative import BiasBaseline, ImplicitALS, ImplicitItemKNN, ItemKNN, MatrixFactorization
from .data import rating_matrix
from .hybrid import DEFAULT_ALPHA, hybrid_scores


def train_test_split(ratings, test_frac=0.2, min_ratings=5, seed=0):
    """У каждого покупателя с min_ratings и более оценок 20% оценок откладываются в тест."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in ratings.groupby("user_idx"):
        if len(g) >= min_ratings:
            n = max(1, int(round(len(g) * test_frac)))
            test_idx.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))
    test = ratings.loc[sorted(test_idx)]
    train = ratings.drop(index=test.index)
    return train, test


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / k


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked, relevant, k):
    """NDCG@k с бинарной релевантностью: DCG = Σ 1/log2(позиция + 1)."""
    rel = set(relevant)
    dcg = sum(1 / np.log2(p + 2) for p, i in enumerate(ranked[:k]) if i in rel)
    idcg = sum(1 / np.log2(p + 2) for p in range(min(k, len(rel))))
    return dcg / idcg if idcg > 0 else 0.0


def ranking_metrics(S, R_train, test, k=10, threshold=4):
    """Метрики ранжирования для матрицы оценок S (users x items).
    Кандидаты - товары, которых нет в обучающей выборке пользователя; релевантны тестовые оценки ≥ threshold."""
    rel_by_user = test[test["rating"] >= threshold].groupby("user_idx")["item_idx"].apply(list)
    item_pop = (R_train > 0).sum(0)
    p_item = item_pop / max(item_pop.sum(), 1)
    P, Rc, N, H, recommended, novelty = [], [], [], [], set(), []
    for u, rel in rel_by_user.items():
        s = np.where(R_train[u] > 0, -np.inf, S[u])
        ranked = list(np.argsort(-s, kind="stable")[:k])
        P.append(precision_at_k(ranked, rel, k))
        Rc.append(recall_at_k(ranked, rel, k))
        N.append(ndcg_at_k(ranked, rel, k))
        H.append(float(len(set(ranked) & set(rel)) > 0))
        recommended.update(ranked)
        novelty.append(np.mean(-np.log2(p_item[ranked] + 1e-9)))
    return {f"Precision@{k}": np.mean(P), f"Recall@{k}": np.mean(Rc), f"NDCG@{k}": np.mean(N),
            f"HitRate@{k}": np.mean(H), "Покрытие каталога": len(recommended) / S.shape[1],
            "Новизна, бит": float(np.mean(novelty))}


def fit_all(R_train, catalog, seed=0):
    """Обучает все модели и возвращает их матрицы оценок."""
    X = content.item_features(catalog)
    base = BiasBaseline().fit(R_train)
    knn = ItemKNN().fit(R_train)
    mf = MatrixFactorization(seed=seed).fit(R_train)
    iknn = ImplicitItemKNN().fit(R_train)
    ials = ImplicitALS(seed=seed).fit(R_train)
    pop = np.tile(((R_train > 0).sum(0) + 0.0), (len(R_train), 1))
    return {
        "Популярность": pop,
        "Средние (μ + bᵤ + bᵢ)": base.predict_all(),
        "Контентная (CB)": content.scores(R_train, X),
        "Item-kNN по оценкам": knn.predict_all(),
        "SVD по оценкам": mf.predict_all(),
        "Item-kNN неявный (CF)": iknn.predict_all(),
        "iALS (CF)": ials.predict_all(),
    }, mf


def cold_start_split(ratings, keep=2, every=4, min_ratings=8, seed=1):
    """Холодный старт: у каждого every-го покупателя с min_ratings+ оценками в обучении остаются
    только keep оценок, остальные уходят в тест. Остальные покупатели целиком в обучении."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for u, g in ratings.groupby("user_idx"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        if len(g) >= min_ratings and u % every == 0:
            train_idx += list(idx[:keep])
            test_idx += list(idx[keep:])
        else:
            train_idx += list(idx)
    return ratings.loc[sorted(train_idx)], ratings.loc[sorted(test_idx)]


def _ranking_models(R_train, catalog, seed=0, alpha=DEFAULT_ALPHA):
    X = content.item_features(catalog)
    cb = content.scores(R_train, X)
    knn = ImplicitItemKNN().fit(R_train).predict_all()
    als = ImplicitALS(seed=seed).fit(R_train).predict_all()
    pop = np.tile(((R_train > 0).sum(0) + 0.0), (len(R_train), 1))
    return {"Популярность": pop, "Контентная (CB)": cb, "Item-kNN неявный (CF)": knn, "iALS (CF)": als,
            "Гибрид": hybrid_scores(knn, als, cb, R_train, alpha)}, (knn, als, cb)


def benchmark(catalog, ratings, n_users, k=10, seed=0, alphas=np.linspace(0, 1, 11)):
    """Полное сравнение моделей. Возвращает словарь таблиц:
    accuracy - точность прогноза оценок (RMSE, MAE);
    ranking  - метрики top-k на отложенных 20% оценок;
    cold     - NDCG@k для «новых» покупателей с 1, 2, 3, 5 оценками;
    alpha    - NDCG@k гибрида при разной доле CF;
    history  - RMSE SVD по эпохам."""
    train, test = train_test_split(ratings, seed=seed)
    R_train = rating_matrix(train, n_users, len(catalog))
    models, mf = fit_all(R_train, catalog, seed)
    models["Гибрид"] = hybrid_scores(models["Item-kNN неявный (CF)"], models["iALS (CF)"],
                                     models["Контентная (CB)"], R_train)

    u, i, y = test["user_idx"].to_numpy(), test["item_idx"].to_numpy(), test["rating"].to_numpy()
    mu = R_train[R_train > 0].mean()
    acc = [{"Модель": "Глобальное среднее", "RMSE": rmse(y, np.full(len(y), mu)), "MAE": mae(y, np.full(len(y), mu))}]
    for name in ["Средние (μ + bᵤ + bᵢ)", "Item-kNN по оценкам", "SVD по оценкам"]:
        p = models[name][u, i]
        acc.append({"Модель": name, "RMSE": rmse(y, p), "MAE": mae(y, p)})
    rank = pd.DataFrame([{"Модель": m, **ranking_metrics(S, R_train, test, k)} for m, S in models.items()])

    knn, als, cb = models["Item-kNN неявный (CF)"], models["iALS (CF)"], models["Контентная (CB)"]
    sweep = [{"α (доля CF)": round(float(a), 2),
              f"NDCG@{k}": ranking_metrics(hybrid_scores(knn, als, cb, R_train, a), R_train, test, k)[f"NDCG@{k}"]}
             for a in alphas]

    cold = []
    for keep in (1, 2, 3, 5):
        tr, te = cold_start_split(ratings, keep=keep, seed=seed + 1)
        Rc = rating_matrix(tr, n_users, len(catalog))
        ms, _ = _ranking_models(Rc, catalog, seed)
        cold.append({"Оценок у нового покупателя": keep,
                     **{m: ranking_metrics(S, Rc, te, k)[f"NDCG@{k}"] for m, S in ms.items()}})
    return {"accuracy": pd.DataFrame(acc), "ranking": rank, "cold": pd.DataFrame(cold),
            "alpha": pd.DataFrame(sweep), "history": mf.history}
