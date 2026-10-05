"""[Member 3] Các model so sánh với XGBoost: Rule-based, Linear Regression, Random Forest, LightGBM.

Mọi model nhận X = prepare_X(df) và có fit(X, y) / predict(X).
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RuleBased:
    """Không học trọng số từng feature: health = exp(-k * mean_i |z_i|), với z_i là độ lệch (theo IQR)
    của feature i so với median của ảnh "khoẻ" (health >= 0.95) trong train. Lệch hai chiều vì
    nhiều feature đi theo cả hai hướng (noise làm gradient tăng, blur làm gradient giảm).
    Chỉ k được chọn trên train (grid search MAE)."""

    def fit(self, X, y):
        healthy = X[y >= 0.95] if (y >= 0.95).sum() >= 10 else X
        self.med = healthy.median()
        self.iqr = (healthy.quantile(0.75) - healthy.quantile(0.25)).replace(0, 1e-6)
        d = self._dist(X)
        self.k = min(np.linspace(0.02, 1, 50), key=lambda k: np.abs(np.exp(-k * d) - y).mean())
        return self

    def _dist(self, X):
        return ((X - self.med).abs() / self.iqr).mean(axis=1).to_numpy()

    def predict(self, X):
        return np.exp(-self.k * self._dist(X))


def get_baselines(seed):
    models = {
        "RuleBased": RuleBased(),
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                              n_jobs=-1, random_state=seed),
    }
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                           random_state=seed, verbose=-1)
    except ImportError:
        print("lightgbm chưa cài -> bỏ qua LightGBM")
    return models
