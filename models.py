import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class SimpleModel:
    def __init__(self):
        self.lang_means = {}   # язык: средний рейтинг
        self.global_mean = 0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SimpleModel":
        data = pd.DataFrame({'lang': X_train['language_code'], 'rating': y_train})
        self.lang_means = data.groupby('lang')['rating'].mean().to_dict()
        self.global_mean = float(y_train.mean())
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return X_test['language_code'].map(self.lang_means).fillna(self.global_mean).values


class RandomForestModel:
    """
    Обёртка над RandomForestRegressor.
    Хранит внутри lang_map, means и stds.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.lang_map: dict = {}
        self.means: pd.Series | None = None
        self.stds:  pd.Series | None = None
        self.numeric_cols = ['num_pages', 'ratings_count', 'text_reviews_count']

    #  Внутренние хелперы

    def _build_lang_map(self, series: pd.Series) -> None:
        unique = series.unique()
        self.lang_map = {str(lang): int(i) for i, lang in enumerate(unique)}

    def _scale_fit(self, X: pd.DataFrame) -> None:
        self.means = X[self.numeric_cols].mean()
        self.stds  = X[self.numeric_cols].std()

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['language_code'] = X['language_code'].map(self.lang_map).fillna(0).astype(int)
        for col in self.numeric_cols:
            X[col] = (X[col] - self.means[col]) / self.stds[col]
        return X

    #  Публичный API

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
        """Обучает модель на необработанных данных."""
        self._build_lang_map(X_train['language_code'])
        self._scale_fit(X_train)
        self.model.fit(self._preprocess(X_train), y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Принимает сырой DataFrame с теми же колонками, что при fit()."""
        return self.model.predict(self._preprocess(X))

    def predict_one(
        self,
        num_pages: float,
        ratings_count: float,
        text_reviews_count: float,
        language_code: str,
    ) -> float:
        row = pd.DataFrame([{
            'num_pages':          num_pages,
            'ratings_count':      ratings_count,
            'text_reviews_count': text_reviews_count,
            'language_code':      language_code,
        }])
        return float(self.predict(row)[0])