"""Рекомендательная система выбора смартфона: K-means, коллаборативная и контентная фильтрация,
гибрид, AHP + TOPSIS + Парето для подбора под бюджет.
Запуск: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from recsys import clustering as cl
from recsys import content
from recsys.advisor import advise, match_segment
from recsys.collaborative import ImplicitALS, ImplicitItemKNN
from recsys.data import CRITERIA, LABELS, criteria_in, index_ratings, load_catalog, rating_matrix, \
    read_csv_any, validate_catalog
from recsys.evaluation import benchmark
from recsys.hybrid import DEFAULT_ALPHA, cf_switching, cluster_popularity, explain_cf, hybrid_scores, \
    to_unit, top_n
from recsys.mcdm import ahp_weights, apply_constraints, matrix_from_upper, pairwise_from_points

DATA = Path(__file__).parent / "data"
ACCENT = "#E07B39"

SCALE = {
    "A намного важнее (9)": 9, "A сильно важнее (7)": 7, "A заметно важнее (5)": 5,
    "A немного важнее (3)": 3, "Одинаково (1)": 1, "B немного важнее (1/3)": 1 / 3,
    "B заметно важнее (1/5)": 1 / 5, "B сильно важнее (1/7)": 1 / 7, "B намного важнее (1/9)": 1 / 9,
}
SCALE_LABELS = list(SCALE)
DEFAULT_POINTS = {"price": 5, "antutu": 3, "camera": 3, "battery": 4, "storage": 3, "ram": 2,
                  "refresh": 2, "weight": 2, "screen": 1, "nfc": 4, "year": 2}
DEFAULT_CRITERIA = ["price", "antutu", "camera", "battery", "storage", "nfc"]


def fmt_rub(x):
    return f"{int(round(x)):,}".replace(",", " ") + " ₽"


# ---------- данные и модели (кэшируются) ----------

@st.cache_data
def load_demo():
    catalog = load_catalog(DATA / "catalog_smartphones.csv")
    users = pd.read_csv(DATA / "users.csv")
    ratings = index_ratings(pd.read_csv(DATA / "ratings.csv"), catalog, users)
    return catalog, users, ratings


@st.cache_resource
def fit_models(_catalog, _ratings, n_users):
    R = rating_matrix(_ratings, n_users, len(_catalog))
    X = content.item_features(_catalog)
    knn = ImplicitItemKNN().fit(R)
    als = ImplicitALS().fit(R)
    return dict(R=R, X=X, sim_cb=X @ X.T, knn=knn, knn_scores=knn.predict_all(),
                als_scores=als.predict_all(), cb_scores=content.scores(R, X))


@st.cache_resource
def fit_clusters(_catalog, _ratings, n_users):
    X_raw, names = cl.behavior_features(_ratings, _catalog, n_users)
    Z, mu, sd = cl.standardize(X_raw)
    best_k, k_table = cl.choose_k(Z)
    return dict(X_raw=X_raw, names=names, Z=Z, mu=mu, sd=sd, best_k=best_k, k_table=k_table)


@st.cache_resource
def kmeans_k(_Z, k):
    labels, centers, _ = cl.kmeans(_Z, k)
    return labels, centers


@st.cache_data
def run_benchmark(_catalog, _ratings, n_users):
    return benchmark(_catalog, _ratings, n_users)


st.set_page_config(page_title="Рекомендательная система: смартфоны", layout="wide")
st.title("Рекомендательная система выбора смартфона")
st.caption("Подбор под бюджет и потребности (AHP + TOPSIS + Парето), сегменты покупателей (K-means), "
           "гибрид коллаборативной и контентной фильтрации")

demo_catalog, users, ratings = load_demo()
n_users = len(users)
models = fit_models(demo_catalog, ratings, n_users)
clus = fit_clusters(demo_catalog, ratings, n_users)

# ---------- боковая панель ----------
with st.sidebar:
    st.header("Каталог")
    source = st.radio("Источник товаров", ["Каталог смартфонов (58 моделей)", "Свой CSV"])
    catalog, own = demo_catalog, False
    if source == "Свой CSV":
        up = st.file_uploader("CSV: model, brand, price и числовые характеристики", type="csv")
        st.caption("Столбцы-критерии: " + ", ".join(CRITERIA) + ". Обязательны model, brand, price.")
        if up is not None:
            try:
                catalog, own = validate_catalog(read_csv_any(up.getvalue())), True
            except Exception as e:
                st.error(f"Не удалось прочитать CSV: {e}")
        else:
            st.info("Пока используется встроенный каталог.")
    st.download_button("Скачать шаблон CSV", (DATA / "catalog_smartphones.csv").read_bytes(),
                       "catalog_template.csv", "text/csv")
    st.divider()
    st.header("Сегменты")
    k = st.slider("Число кластеров K-means", 2, 8, clus["best_k"],
                  help=f"По максимуму силуэта лучшее k = {clus['best_k']}")
    labels, centers = kmeans_k(clus["Z"], k)
    profiles = cl.describe_clusters(labels, clus["X_raw"], clus["names"], ratings, demo_catalog)
    seg_names = {int(r["Кластер"]): cl.cluster_name(r) for _, r in profiles.iterrows()}
    seg_pop = cluster_popularity(ratings, labels, len(demo_catalog))
    st.caption(f"Покупателей: {n_users}, оценок: {len(ratings)}, товаров: {len(demo_catalog)}")

tab_pick, tab_seg, tab_user, tab_quality, tab_cat = st.tabs(
    ["Подбор под бюджет", "Сегменты покупателей", "Персональные рекомендации", "Качество моделей", "Каталог"])

# ================= 1. Подбор под бюджет (новый покупатель) =================
def pick_tab():
    st.subheader("Что вам нужно от смартфона?")
    crit_all = criteria_in(catalog)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    pmin, pmax = int(catalog["price"].min()), int(catalog["price"].max())
    budget = c1.slider("Бюджет, руб.", pmin, pmax, (min(max(pmin, 15000), pmax), min(40000, pmax)), step=1000,
                       key="budget")
    need_nfc = c2.checkbox("Обязательно NFC (оплата телефоном)", True) if "nfc" in catalog else False
    min_ram = c2.select_slider("Оперативная память не меньше, ГБ", [0, 4, 6, 8, 12], 6) if "ram" in catalog else 0
    min_storage = c3.select_slider("Встроенная память не меньше, ГБ", [0, 64, 128, 256, 512], 128) \
        if "storage" in catalog else 0
    exclude = c3.multiselect("Не рассматривать бренды", sorted(catalog["brand"].unique()))
    allowed = apply_constraints(catalog, budget, need_nfc, min_ram, min_storage, exclude)
    st.caption(f"Под ограничения подходит {int(allowed.sum())} из {len(catalog)} моделей.")

    st.markdown("**Важность критериев**")
    chosen = st.multiselect("Какие критерии учитывать", crit_all,
                            default=[c for c in DEFAULT_CRITERIA if c in crit_all] or crit_all[:4],
                            format_func=lambda c: LABELS[c])
    if len(chosen) < 2:
        st.warning("Выберите минимум два критерия.")
        return
    mode = st.radio("Как задать важность", ["Баллы от 1 до 5", "Попарные сравнения по шкале Саати"],
                    horizontal=True, key="weight_mode")
    points = [DEFAULT_POINTS.get(c, 3) for c in chosen]
    if mode.startswith("Баллы"):
        cols = st.columns(min(len(chosen), 6))
        points = [cols[i % len(cols)].slider(LABELS[c], 1, 5, points[i], key=f"pt_{c}") for i, c in enumerate(chosen)]
        M = pairwise_from_points(points)
    else:
        start = pairwise_from_points(points)
        nearest = lambda v: min(SCALE_LABELS, key=lambda s: abs(np.log(SCALE[s]) - np.log(v)))
        upper = {}
        with st.expander("Матрица попарных сравнений", expanded=True):
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    lab = st.select_slider(f"{LABELS[chosen[i]]} (A)  vs  {LABELS[chosen[j]]} (B)", SCALE_LABELS,
                                           value=nearest(start[i, j]), key=f"pw_{chosen[i]}_{chosen[j]}")
                    upper[(i, j)] = SCALE[lab]
        M = matrix_from_upper(len(chosen), upper)
    w, lam, ci, cr = ahp_weights(M)

    wc1, wc2 = st.columns([1, 1.3])
    with wc1:
        m1, m2, m3 = st.columns(3)
        m1.metric("λ max", f"{lam:.3f}")
        m2.metric("CI", f"{ci:.3f}")
        m3.metric("CR", f"{cr:.3f}")
        if cr < 0.1:
            st.success("CR < 0,1: суждения согласованы.")
        else:
            st.error("CR ≥ 0,1: суждения противоречивы, пересмотрите сравнения.")
        beta = st.slider("Учитывать опыт похожих покупателей (β)", 0.0, 1.0, 0.3, 0.05, disabled=own,
                         help="0 - только ваши критерии (TOPSIS), 1 - только то, что хвалят в вашем сегменте")
    with wc2:
        wdf = pd.DataFrame({"Критерий": [LABELS[c] for c in chosen], "Вес": w}).sort_values("Вес")
        fig = px.bar(wdf, x="Вес", y="Критерий", orientation="h", text_auto=".3f", title="Веса критериев (AHP)",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)

    if allowed.sum() == 0:
        st.error("Под заданные ограничения не подходит ни одна модель. Расширьте бюджет или ослабьте требования.")
        return

    seg_scores, seg_id = None, None
    if not own:
        prelim = advise(catalog, chosen, w, allowed, None, 0.0, n=5)
        seg_id, _ = match_segment(catalog, prelim["index"].tolist(), clus["X_raw"], clus["mu"], clus["sd"], centers)
        seg_scores = seg_pop[seg_id]
    result = advise(catalog, chosen, w, allowed, seg_scores, beta if not own else 0.0, n=None)
    best = result.iloc[0]
    st.success(f"Рекомендация: **{best['model']}** за {fmt_rub(best['price'])}. "
               f"Сильные стороны: {best['Сильные стороны'].lower()}.")
    if seg_id is not None:
        st.info(f"По вашим требованиям вы ближе всего к сегменту {seg_id}: {seg_names[seg_id]}. "
                f"Покупатели этого сегмента чаще всего хвалят: {profiles.loc[seg_id, 'Чаще всего хвалят']}.")

    show = result.head(10)[["Место", "model", "price", "TOPSIS", "Итог", "Парето", "Сильные стороны"] +
                           (["Сегмент (0..1)"] if "Сегмент (0..1)" in result else [])].copy()
    show["Парето"] = show["Парето"].map({True: "да", False: ""})
    show = show.rename(columns={"model": "Модель", "price": "Цена, руб."})
    st.dataframe(show.round(3), hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        sc = result.copy()
        sc["Группа"] = np.where(sc["Место"] <= 3, "Топ-3", np.where(sc["Парето"], "Парето-оптимальная", "Доминируемая"))
        fig = px.scatter(sc, x="price", y="TOPSIS", color="Группа", hover_name="model", symbol="Группа",
                         color_discrete_map={"Топ-3": ACCENT, "Парето-оптимальная": "#1D2A40", "Доминируемая": "#B8C2CF"},
                         labels={"price": "Цена, руб.", "TOPSIS": "Близость к идеалу C"},
                         title="Цена и соответствие требованиям")
        st.plotly_chart(fig)
    with g2:
        top3 = result.head(3)
        fig = go.Figure()
        for _, row in top3.iterrows():
            vals = []
            for c in chosen:
                x = catalog.loc[allowed, c].astype(float)
                lo, hi = x.min(), x.max()
                v = 0.5 if hi == lo else (row[c] - lo) / (hi - lo)
                vals.append(1 - v if CRITERIA[c][2] == "min" else v)
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[LABELS[c] for c in chosen] + [LABELS[chosen[0]]],
                                          fill="toself", name=row["model"]))
        fig.update_layout(title="Топ-3: сравнение по критериям (1 = лучшее среди допустимых)",
                          polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(fig)
    st.download_button("Скачать подборку (CSV)",
                       result.drop(columns=["index"]).to_csv(index=False).encode("utf-8-sig"),
                       "recommendations.csv", "text/csv")


with tab_pick:
    pick_tab()

# ================= 2. Сегменты покупателей =================
with tab_seg:
    st.subheader("Кластеризация покупателей по покупательскому поведению (K-means)")
    st.caption("Признаки покупателя: цена понравившихся моделей, средняя оценка, число оценок, доля премиальных "
               "брендов и средние характеристики понравившихся моделей. Признаки стандартизованы (z-оценки).")
    s1, s2 = st.columns(2)
    kt = clus["k_table"]
    with s1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kt["k"], y=kt["Силуэт"], mode="lines+markers", name="Силуэт", line=dict(color=ACCENT)))
        fig.add_trace(go.Scatter(x=kt["k"], y=kt["Инерция"], mode="lines+markers", name="Инерция", yaxis="y2",
                                 line=dict(color="#1D2A40")))
        fig.update_layout(title="Выбор числа кластеров: силуэт и метод локтя", xaxis_title="k",
                          yaxis=dict(title="Силуэт"), yaxis2=dict(title="Инерция", overlaying="y", side="right"))
        st.plotly_chart(fig)
    with s2:
        P, ev = cl.pca2(clus["Z"])
        pdf = pd.DataFrame({"ГК 1": P[:, 0], "ГК 2": P[:, 1], "Кластер": labels.astype(str),
                            "Бюджет": users["budget"]})
        fig = px.scatter(pdf, x="ГК 1", y="ГК 2", color="Кластер", opacity=0.7,
                         title=f"Покупатели на двух главных компонентах ({ev.sum():.0%} дисперсии)")
        st.plotly_chart(fig)
    st.markdown(f"**Профили кластеров (k = {k}, силуэт = {cl.silhouette(clus['Z'], labels):.3f})**")
    prof = profiles.copy()
    prof.insert(1, "Описание", prof["Кластер"].map(seg_names))
    st.dataframe(prof, hide_index=True)
    heat = pd.DataFrame(centers, columns=clus["names"], index=[f"Кластер {i}" for i in range(k)])
    fig = px.imshow(heat, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                    title="Центры кластеров (z-оценки): красное - выше среднего, синее - ниже")
    st.plotly_chart(fig)
    with st.expander("Проверка: совпадают ли кластеры со скрытыми сегментами генератора"):
        st.write(f"Скорректированный индекс Рэнда (ARI) = {cl.adjusted_rand_index(labels, users['segment']):.3f} "
                 "(1 - полное совпадение, 0 - случайное разбиение). Скрытый сегмент методам не передаётся.")
        st.dataframe(pd.crosstab(pd.Series(labels, name="Кластер"), users["segment"].rename("Скрытый сегмент")))

# ================= 3. Персональные рекомендации =================
with tab_user:
    st.subheader("Рекомендации покупателю с историей покупок")
    R = models["R"]
    counts = (R > 0).sum(1)
    u1, u2, u3 = st.columns([1, 1, 1.2])
    uid = u1.number_input("ID покупателя", 1, n_users, 7)
    u = int(uid) - 1
    alpha = u2.slider("α: доля коллаборативной части", 0.0, 1.0, DEFAULT_ALPHA, 0.05,
                      help="Гибрид = α·CF + (1-α)·CB. α = 0: только похожесть характеристик, α = 1: только поведение покупателей")
    in_budget = u3.checkbox("Только в привычном ценовом диапазоне покупателя", True,
                            help="Не дороже 1,3 × самой дорогой модели, которую покупатель оценил на 4-5")
    hist = demo_catalog.loc[R[u] > 0, ["model", "price"]].assign(Оценка=R[u][R[u] > 0])
    liked_max = hist.loc[hist["Оценка"] >= 4, "price"].max() if (hist["Оценка"] >= 4).any() else hist["price"].max()
    allowed_u = (demo_catalog["price"] <= 1.3 * liked_max).to_numpy() if in_budget else None
    st.write(f"Оценок: **{counts[u]}**, сегмент: **{labels[u]}** ({seg_names[int(labels[u])]}). "
             f"Режим CF: {'item-kNN + iALS' if counts[u] >= 5 else 'только item-kNN (мало истории)'}.")
    with st.expander("История оценок покупателя", expanded=False):
        st.dataframe(hist.sort_values("Оценка", ascending=False).rename(columns={"model": "Модель", "price": "Цена, руб."}),
                     hide_index=True)

    cf = cf_switching(models["knn_scores"], models["als_scores"], R)
    cb = to_unit(models["cb_scores"], R)
    hy = hybrid_scores(models["knn_scores"], models["als_scores"], models["cb_scores"], R, alpha)
    rated = R[u] > 0
    lists = {"Коллаборативная (CF)": top_n(cf[u], rated, allowed_u), "Контентная (CB)": top_n(cb[u], rated, allowed_u),
             "Гибрид": top_n(hy[u], rated, allowed_u)}
    cols = st.columns(3)
    for col, (name, idx) in zip(cols, lists.items()):
        col.markdown(f"**{name}**")
        col.dataframe(pd.DataFrame({"Модель": demo_catalog["model"].iloc[idx].values,
                                    "Цена": demo_catalog["price"].iloc[idx].map(fmt_rub).values}), hide_index=True)
    idx = lists["Гибрид"]
    if idx:
        parts = pd.DataFrame({"Модель": demo_catalog["model"].iloc[idx].values,
                              "CF · α": alpha * cf[u, idx], "CB · (1-α)": (1 - alpha) * cb[u, idx]})
        fig = px.bar(parts.melt(id_vars="Модель", var_name="Часть", value_name="Вклад"), y="Модель", x="Вклад",
                     color="Часть", orientation="h", title="Из чего складывается оценка гибрида",
                     color_discrete_map={"CF · α": "#1D2A40", "CB · (1-α)": ACCENT})
        fig.update_yaxes(autorange="reversed")
        g1, g2 = st.columns([1.2, 1])
        g1.plotly_chart(fig)
        expl = [explain_cf(u, i, R, models["knn"].S, demo_catalog) for i in idx[:5]]
        g2.markdown("**Почему рекомендовано (топ-5)**")
        for i, e in zip(idx[:5], expl):
            g2.write(f"• **{demo_catalog['model'].iloc[i]}**: {e or 'похож по характеристикам на ваши покупки'}")

# ================= 4. Качество моделей =================
with tab_quality:
    st.subheader("Оценка качества на отложенной выборке")
    st.caption("У каждого покупателя с 5+ оценками 20% оценок спрятаны. Модель ранжирует все ещё не оценённые товары; "
               "релевантными считаются спрятанные товары с оценкой 4-5. Метрики усреднены по покупателям.")
    with st.spinner("Обучаю модели и считаю метрики..."):
        bm = run_benchmark(demo_catalog, ratings, n_users)
    rk = bm["ranking"]
    st.dataframe(rk.round(3), hide_index=True)
    q1, q2 = st.columns(2)
    with q1:
        long = rk.melt(id_vars="Модель", value_vars=["Precision@10", "Recall@10", "NDCG@10", "HitRate@10"],
                       var_name="Метрика", value_name="Значение")
        fig = px.bar(long, x="Значение", y="Модель", color="Метрика", barmode="group", orientation="h",
                     title="Метрики ранжирования (top-10)")
        st.plotly_chart(fig)
    with q2:
        cold = bm["cold"].melt(id_vars="Оценок у нового покупателя", var_name="Модель", value_name="NDCG@10")
        fig = px.line(cold, x="Оценок у нового покупателя", y="NDCG@10", color="Модель", markers=True,
                      title="Холодный старт: качество при 1-5 оценках")
        st.plotly_chart(fig)
    q3, q4 = st.columns(2)
    with q3:
        fig = px.line(bm["alpha"], x="α (доля CF)", y="NDCG@10", markers=True,
                      title="Чувствительность гибрида к α", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig)
    with q4:
        st.markdown("**Точность прогноза оценки (RMSE, MAE; меньше = лучше)**")
        st.dataframe(bm["accuracy"].round(4), hide_index=True)
        fig = px.line(x=np.arange(1, len(bm["history"]) + 1), y=bm["history"], markers=False,
                      labels={"x": "Эпоха", "y": "RMSE на обучении"}, title="Обучение SVD (SGD)")
        fig.update_layout(height=260, margin=dict(t=40, b=10))
        st.plotly_chart(fig)
    best_row = rk.loc[rk["NDCG@10"].idxmax()]
    st.info(f"Лучшая модель по NDCG@10: **{best_row['Модель']}** ({best_row['NDCG@10']:.3f}). "
            "Модели, предсказывающие величину оценки (SVD, kNN по оценкам), плохо ранжируют: важнее угадать, "
            "что покупатель вообще выберет, поэтому для top-N используется неявная обратная связь.")

# ================= 5. Каталог =================
with tab_cat:
    st.subheader("Каталог")
    st.dataframe(catalog, hide_index=True)
    if "antutu" in catalog:
        fig = px.scatter(catalog, x="price", y="antutu", color="brand", hover_name="model", log_x=True,
                         labels={"price": "Цена, руб. (лог. шкала)", "antutu": "AnTuTu, тыс.", "brand": "Бренд"},
                         title="Цена и производительность")
        st.plotly_chart(fig)
    if not own:
        pop = ratings.groupby("item_idx")["rating"].agg(["mean", "count"]).reset_index()
        pop["Модель"] = demo_catalog["model"].iloc[pop["item_idx"]].values
        fig = px.scatter(pop, x="count", y="mean", hover_name="Модель", labels={"count": "Число оценок",
                         "mean": "Средняя оценка"}, title="Популярность и средняя оценка моделей",
                         color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig)
