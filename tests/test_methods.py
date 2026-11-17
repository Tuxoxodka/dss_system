from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys import clustering as cl
from recsys import content
from recsys.advisor import advise, match_segment
from recsys.collaborative import BiasBaseline, ImplicitALS, ImplicitItemKNN, ItemKNN, MatrixFactorization
from recsys.data import index_ratings, load_catalog, minmax_benefit, rating_matrix, read_csv_any, validate_catalog
from recsys.evaluation import benchmark, ndcg_at_k, precision_at_k, ranking_metrics, recall_at_k, train_test_split
from recsys.hybrid import blend, cf_switching, cluster_popularity, hybrid_scores, to_unit, top_n
from recsys.mcdm import ahp_weights, apply_constraints, matrix_from_upper, pairwise_from_points, pareto_front, topsis
from recsys.simulate import generate

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def demo():
    catalog = load_catalog(DATA / "catalog_smartphones.csv")
    users = pd.read_csv(DATA / "users.csv")
    ratings = index_ratings(pd.read_csv(DATA / "ratings.csv"), catalog, users)
    return catalog, users, ratings


# ---------- данные ----------

def test_catalog_is_valid(demo):
    catalog, users, ratings = demo
    assert len(catalog) == 58 and catalog["model"].is_unique
    assert ratings["rating"].between(1, 5).all()
    assert len(ratings) == len(pd.read_csv(DATA / "ratings.csv"))       # все оценки сопоставлены


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"model": ["A", "B"], "brand": ["x", "y"]}),                              # нет цены
    pd.DataFrame({"model": ["A", "A"], "brand": ["x", "y"], "price": [1, 2], "ram": [1, 2], "nfc": [1, 0]}),
    pd.DataFrame({"model": ["A", "B"], "brand": ["x", "y"], "price": [0, 2], "ram": [1, 2], "nfc": [1, 0]}),
    pd.DataFrame({"model": ["A", "B"], "brand": ["x", "y"], "price": [1, 2], "ram": [1, "?"], "nfc": [1, 0]}),
    pd.DataFrame({"model": ["A"], "brand": ["x"], "price": [1], "ram": [1], "nfc": [1]}),
])
def test_validate_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate_catalog(bad)


def test_read_csv_russian_excel():
    raw = "model;brand;price;screen;ram\nA;x;100;6,5;8\nB;y;200;6,1;4\n".encode("cp1251")
    df = validate_catalog(read_csv_any(raw))
    assert df["screen"].tolist() == [6.5, 6.1] and "item_id" in df


def test_minmax_benefit_flips_min_criteria():
    df = pd.DataFrame({"price": [10, 20, 30], "ram": [4, 8, 12]})
    F = minmax_benefit(df, ["price", "ram"])
    assert F[:, 0].tolist() == [1, 0.5, 0] and F[:, 1].tolist() == [0, 0.5, 1]


def test_generator_is_reproducible(demo):
    catalog = demo[0]
    u1, r1 = generate(catalog, n_users=50, seed=1)
    u2, r2 = generate(catalog, n_users=50, seed=1)
    assert r1.equals(r2) and u1.equals(u2)


# ---------- K-means ----------

def test_kmeans_finds_separated_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, (40, 2)) for c in ([0, 0], [5, 5], [0, 5])])
    truth = np.repeat([0, 1, 2], 40)
    labels, centers, _ = cl.kmeans(X, 3)
    assert cl.adjusted_rand_index(labels, truth) == pytest.approx(1)
    best_k, table = cl.choose_k(X, range(2, 6))
    assert best_k == 3 and table["Силуэт"].max() > 0.7


def test_silhouette_and_ari_edge_cases():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    assert cl.silhouette(X, [0, 0, 1, 1]) > 0.9
    assert cl.adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1)


def test_clusters_reflect_hidden_segments(demo):
    catalog, users, ratings = demo
    X, _ = cl.behavior_features(ratings, catalog, len(users))
    Z, _, _ = cl.standardize(X)
    labels, _, _ = cl.kmeans(Z, 4)
    # скрытый сегмент методам не передаётся, но кластеры должны заметно с ним совпадать
    assert cl.adjusted_rand_index(labels, users["segment"]) > 0.3


# ---------- коллаборативная и контентная фильтрация ----------

R_SMALL = np.array([[5, 4, 0, 1],
                    [4, 5, 1, 0],
                    [1, 0, 5, 4],
                    [0, 1, 4, 5],
                    [5, 0, 0, 1]], dtype=float)


def test_item_knn_rating_similar_items():
    knn = ItemKNN(k=3, shrink=0).fit(R_SMALL)
    assert knn.S[0, 1] > 0 > knn.S[0, 2]                     # 0 и 1 любят одни и те же люди
    P = knn.predict_all()
    assert P[4, 1] > P[4, 2]                                 # пользователю 4 понравится товар 1, а не 2
    assert P.min() >= 1 and P.max() <= 5


R_BLOCK = np.array([[5, 4, 0, 0], [4, 5, 0, 0], [5, 5, 0, 0],
                    [0, 0, 5, 4], [0, 0, 4, 5], [0, 0, 5, 5],
                    [5, 0, 0, 0]], dtype=float)


def test_implicit_models_rank_cooccurring_item():
    # последний покупатель взял товар 0; с ним всегда покупают товар 1, а не 2 или 3
    for model in (ImplicitItemKNN(shrink=0), ImplicitALS(factors=2, iters=20)):
        S = model.fit(R_BLOCK).predict_all()
        assert S[6, 1] > S[6, 2] and S[6, 1] > S[6, 3]


def test_matrix_factorization_learns():
    mf = MatrixFactorization(factors=2, epochs=200, lr=0.02, reg=0.01).fit(R_SMALL)
    assert mf.history[-1] < mf.history[0] and mf.history[-1] < 0.3
    base = BiasBaseline().fit(R_SMALL).predict_all()
    assert base.shape == R_SMALL.shape


def test_content_profile_prefers_similar_items(demo):
    catalog = demo[0]
    X = content.item_features(catalog)
    R = np.zeros((1, len(catalog)))
    flagship = catalog.index[catalog["model"].str.contains("S25 Ultra")][0]
    budget = catalog.index[catalog["model"].str.contains("Redmi A3")][0]
    R[0, flagship], R[0, budget] = 5, 1
    s = content.scores(R, X)[0]
    iphone = catalog.index[catalog["model"].str.contains("16 Pro Max")][0]
    assert s[iphone] > s[catalog.index[catalog["model"].str.contains("Redmi 14C")][0]]


# ---------- гибрид ----------

def test_to_unit_and_blend_extremes():
    R = np.array([[5, 0, 0]], dtype=float)
    cf = np.array([[9, 2, 1]], dtype=float)
    cb = np.array([[0, 0.1, 0.9]])
    assert to_unit(cf, R)[0].tolist() == [1, 1, 0]           # нормировка по неоценённым товарам
    assert top_n(blend(cf, cb, R, alpha=1)[0], R[0] > 0) == [1, 2]
    assert top_n(blend(cf, cb, R, alpha=0)[0], R[0] > 0) == [2, 1]


def test_switching_uses_knn_for_short_history():
    R = np.array([[5, 0, 0, 0], [5, 4, 3, 5]], dtype=float)
    knn = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
    als = np.array([[0, 0, 1, 0], [0, 0, 0, 0]], dtype=float)
    cf = cf_switching(knn, als, R, min_ratings=4)
    assert cf[0, 1] == 1 and cf[0, 2] == 0


def test_top_n_respects_filters():
    assert top_n([5, 4, 3, 2], exclude=[True, False, False, False], allowed=[True, True, False, True], n=5) == [1, 3]


def test_cluster_popularity_bayesian(demo):
    catalog, users, ratings = demo
    labels = np.zeros(len(users), dtype=int)
    pop = cluster_popularity(ratings, labels, len(catalog))
    most = ratings.groupby("item_idx").size().idxmax()
    never = set(range(len(catalog))) - set(ratings["item_idx"])
    assert pop.shape == (1, len(catalog)) and pop[0, most] > np.median(pop[0])
    assert all(pop[0, i] == 0 for i in never)


# ---------- AHP, TOPSIS, Парето ----------

def test_ahp_matches_teacher_example_4x4():
    # Этап 1 из файла преподавателя «Метод Саати.xls», лист МАИ
    M = [[1, 1, 3, 5], [1, 1, 3, 5], [1 / 3, 1 / 3, 1, 3], [1 / 5, 1 / 5, 1 / 3, 1]]
    w, lam, ci, cr = ahp_weights(M)
    assert np.allclose(w, [0.390813, 0.390813, 0.150892, 0.067481], atol=1e-6)
    assert cr == pytest.approx(0.011629, abs=1e-6)


def test_pairwise_from_points_and_upper():
    M = pairwise_from_points([5, 4, 2, 1])
    assert np.allclose(M * M.T, 1) and M[0, 1] == 3 and M[0, 3] == 9
    assert np.allclose(matrix_from_upper(3, {(0, 1): 3, (0, 2): 5, (1, 2): 3}) @ np.ones(3) > 0, True)
    assert ahp_weights(pairwise_from_points([3, 3, 3]))[3] == pytest.approx(0)


def test_topsis_and_pareto():
    X = np.array([[10, 9], [20, 3], [15, 6], [12, 9]], dtype=float)
    c, _, _ = topsis(X, ["min", "max"], [0.5, 0.5])
    assert c[0] == pytest.approx(1) and c[1] == pytest.approx(0)
    assert pareto_front(X, ["min", "max"]).tolist() == [True, False, False, False]


def test_constraints(demo):
    catalog = demo[0]
    ok = apply_constraints(catalog, (10000, 30000), need_nfc=True, min_ram=8, exclude_brands=("POCO",))
    sub = catalog[ok]
    assert len(sub) > 0 and sub["price"].between(10000, 30000).all() and (sub["ram"] >= 8).all()
    assert (sub["nfc"] == 1).all() and "POCO" not in set(sub["brand"])


def test_advise_and_segment_match(demo):
    catalog, users, ratings = demo
    crit = ["price", "antutu", "battery"]
    w, *_ = ahp_weights(pairwise_from_points([3, 5, 3]))
    allowed = apply_constraints(catalog, (20000, 60000))
    res = advise(catalog, crit, w, allowed, None, 0.0, n=None)
    assert res["Место"].tolist() == list(range(1, len(res) + 1))
    assert res.iloc[0]["model"] == res.sort_values("TOPSIS").iloc[-1]["model"]   # β = 0 -> чистый TOPSIS
    X, _ = cl.behavior_features(ratings, catalog, len(users))
    Z, mu, sd = cl.standardize(X)
    _, centers, _ = cl.kmeans(Z, 3)
    seg, d = match_segment(catalog, res["index"].head(5), X, mu, sd, centers)
    assert 0 <= seg < 3 and d[seg] == d.min()


# ---------- метрики и сравнение моделей ----------

def test_ranking_metrics_by_hand():
    ranked, rel = [3, 1, 7, 2], [1, 2, 9]
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(ranked, rel, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([1, 2], [1, 2], 2) == pytest.approx(1)


def test_split_keeps_every_user(demo):
    _, users, ratings = demo
    train, test = train_test_split(ratings)
    assert len(train) + len(test) == len(ratings)
    assert set(train["user_idx"]) == set(ratings["user_idx"])
    assert set(test.index).isdisjoint(train.index)


def test_benchmark_hybrid_beats_baselines(demo):
    catalog, users, ratings = demo
    res = benchmark(catalog, ratings, len(users), alphas=[0.0, 0.9, 1.0])
    rk = res["ranking"].set_index("Модель")["NDCG@10"]
    assert rk["Гибрид"] == rk.max()
    assert rk["Гибрид"] > rk["Популярность"] and rk["Гибрид"] > rk["Контентная (CB)"]
    cold = res["cold"].set_index("Оценок у нового покупателя")
    assert (cold["Гибрид"] >= cold["Популярность"]).all()
    acc = res["accuracy"].set_index("Модель")["RMSE"]
    assert acc["SVD по оценкам"] < acc["Глобальное среднее"]
