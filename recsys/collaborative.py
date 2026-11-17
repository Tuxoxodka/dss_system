"""Коллаборативная фильтрация: базовые смещения, item-based kNN, матричная факторизация (SVD с смещениями)."""
import numpy as np


class BiasBaseline:
    """r̂ = μ + b_u + b_i (смещения с регуляризацией)."""

    def __init__(self, reg=5.0):
        self.reg = reg

    def fit(self, R):
        mask = R > 0
        self.mu = R[mask].mean()
        self.bi = ((R - self.mu) * mask).sum(0) / (mask.sum(0) + self.reg)
        resid = (R - self.mu - self.bi[None]) * mask
        self.bu = resid.sum(1) / (mask.sum(1) + self.reg)
        return self

    def predict_all(self):
        return np.clip(self.mu + self.bu[:, None] + self.bi[None], 1, 5)


class ItemKNN:
    """Item-based коллаборативная фильтрация.
    Сходство товаров - скорректированный косинус (оценки центрируются средним пользователя),
    с поправкой на число общих оценщиков (shrinkage).
    Прогноз: r̂_ui = μ_u + Σ s_ij (r_uj - μ_u) / Σ |s_ij| по k самым похожим товарам, которые оценил u."""

    def __init__(self, k=15, shrink=10.0):
        self.k, self.shrink = k, shrink

    def fit(self, R):
        self.R = R
        self.mask = R > 0
        cnt = self.mask.sum(1)
        self.user_mean = np.divide(R.sum(1), cnt, out=np.full(len(R), R[self.mask].mean()), where=cnt > 0)
        C = (R - self.user_mean[:, None]) * self.mask
        num = C.T @ C
        norm = np.sqrt((C ** 2).sum(0))
        S = num / (np.outer(norm, norm) + 1e-9)
        co = self.mask.T.astype(float) @ self.mask.astype(float)   # число общих оценщиков
        S = S * co / (co + self.shrink)
        np.fill_diagonal(S, 0)
        self.S = S
        self.baseline = BiasBaseline().fit(R)
        return self

    def predict_all(self):
        R, mask, S = self.R, self.mask, self.S
        n_users, n_items = R.shape
        P = self.baseline.predict_all().copy()
        C = (R - self.user_mean[:, None]) * mask
        for u in range(n_users):
            rated = np.where(mask[u])[0]
            if len(rated) == 0:
                continue
            sims = S[:, rated]                                    # items x rated
            if len(rated) > self.k:                               # оставляем k ближайших соседей
                thr = -np.sort(-np.abs(sims), axis=1)[:, self.k - 1:self.k]
                sims = np.where(np.abs(sims) >= thr, sims, 0)
            den = np.abs(sims).sum(1)
            num = sims @ C[u, rated]
            ok = den > 1e-6
            P[u, ok] = self.user_mean[u] + num[ok] / den[ok]
        return np.clip(P, 1, 5)

    def similar_items(self, i, n=5):
        order = np.argsort(-self.S[i])
        return order[:n], self.S[i, order[:n]]


class ImplicitItemKNN:
    """Item-based CF по неявной обратной связи (факт покупки/оценки, а не её величина).
    Сходство: косинус векторов «кто купил» с поправкой на число общих покупателей.
    Оценка товара i для u: Σ_j∈I_u s_ij · (r_uj / 5) - чем больше похожих на i товаров купил
    и похвалил пользователь, тем выше i."""

    def __init__(self, shrink=10.0):
        self.shrink = shrink

    def fit(self, R):
        self.R = R
        B = (R > 0).astype(float)
        co = B.T @ B
        norm = np.sqrt(np.diag(co))
        cos = co / (np.outer(norm, norm) + 1e-9)
        S = cos * np.divide(co, co + self.shrink, out=np.zeros_like(co), where=(co + self.shrink) > 0)
        np.fill_diagonal(S, 0)
        self.S = S
        return self

    def predict_all(self):
        return (self.R / 5.0) @ self.S.T

    def similar_items(self, i, n=5):
        order = np.argsort(-self.S[i])
        return order[:n], self.S[i, order[:n]]


class ImplicitALS:
    """Матричная факторизация для неявной обратной связи (Hu, Koren, Volinsky, 2008).
    Предпочтение p_ui = 1, если товар оценён, иначе 0; доверие c_ui = 1 + a · r_ui.
    Минимизируется Σ c_ui (p_ui - x_u·y_i)² + λ(|x|² + |y|²) поочерёдно по X и Y (ALS)."""

    def __init__(self, factors=8, reg=0.1, alpha=2.0, iters=15, seed=0):
        self.factors, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, Fixed, Cmat, Pmat):
        k = self.factors
        FtF = Fixed.T @ Fixed
        out = np.zeros((Cmat.shape[0], k))
        for u in range(Cmat.shape[0]):
            c = Cmat[u]
            A = FtF + (Fixed.T * (c - 1)) @ Fixed + self.reg * np.eye(k)
            b = (Fixed.T * c) @ Pmat[u]
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, R):
        rng = np.random.default_rng(self.seed)
        P = (R > 0).astype(float)
        C = 1 + self.alpha * R
        self.X = rng.normal(0, 0.1, (R.shape[0], self.factors))
        self.Y = rng.normal(0, 0.1, (R.shape[1], self.factors))
        self.history = []
        for _ in range(self.iters):
            self.X = self._solve(self.Y, C, P)
            self.Y = self._solve(self.X, C.T, P.T)
            self.history.append(float((C * (P - self.X @ self.Y.T) ** 2).sum()))
        return self

    def predict_all(self):
        return self.X @ self.Y.T


class MatrixFactorization:
    """Матричная факторизация (Funk SVD со смещениями), обучение стохастическим градиентным спуском:
    r̂_ui = μ + b_u + b_i + p_u · q_i,   минимизируется Σ (r - r̂)² + λ (b² + |p|² + |q|²)."""

    def __init__(self, factors=8, epochs=40, lr=0.01, reg=0.05, seed=0):
        self.factors, self.epochs, self.lr, self.reg, self.seed = factors, epochs, lr, reg, seed

    def fit(self, R):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape
        users, items = np.nonzero(R)
        vals = R[users, items]
        self.mu = vals.mean()
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.P = rng.normal(0, 0.1, (n_users, self.factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.factors))
        self.history = []
        lr, reg = self.lr, self.reg
        order = np.arange(len(vals))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for t in order:
                u, i, r = users[t], items[t], vals[t]
                pu = self.P[u].copy()
                e = r - (self.mu + self.bu[u] + self.bi[i] + pu @ self.Q[i])
                self.bu[u] += lr * (e - reg * self.bu[u])
                self.bi[i] += lr * (e - reg * self.bi[i])
                self.P[u] += lr * (e * self.Q[i] - reg * pu)
                self.Q[i] += lr * (e * pu - reg * self.Q[i])
            pred = self.mu + self.bu[users] + self.bi[items] + (self.P[users] * self.Q[items]).sum(1)
            self.history.append(float(np.sqrt(((vals - pred) ** 2).mean())))
        return self

    def predict_all(self):
        return np.clip(self.mu + self.bu[:, None] + self.bi[None] + self.P @ self.Q.T, 1, 5)
