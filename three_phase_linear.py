"""
Модуль для багатофазного прогнозування часових рядів.
Використовує комбінацію статистичного бейзлайну та градієнтного бустингу (XGBoost).
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelSettings:
    """Конфігурація параметрів моделі"""
    date_idx: str
    target: str
    groups: List[str]
    horizon: int  # глибина прогнозу
    frequency: str = 'MS'
    min_obs: int = 15
    lags: List[int] = (1, 3, 6, 12)
    windows: List[int] = (3, 6)
    n_cv_splits: int = 3
    seed: int = 42

class TurnoverPredictor:
    def __init__(self, config: ModelSettings):
        self.cfg = config
        self.features: List[str] = []

    def _build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фаза 2: Генерація ознак часу та циклічності"""
        dates = pd.to_datetime(df[self.cfg.date_idx])
        df['idx_month'] = dates.dt.month
        df['idx_quarter'] = dates.dt.quarter
        df['idx_year'] = dates.dt.year
        
        # Циклічні ознаки (Sin/Cos трансформація)
        df['month_sin'] = np.sin(2 * np.pi * df['idx_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['idx_month'] / 12)
        return df

    def _build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фаза 2: Лаги та ковзні середні"""
        for l in self.cfg.lags:
            df[f'prev_val_{l}'] = df[self.cfg.target].shift(l)
        
        for w in self.cfg.windows:
            df[f'roll_avg_{w}'] = df[self.cfg.target].shift(1).rolling(window=w).mean()
        return df

    def _get_statistical_baseline(self, series: pd.Series) -> np.ndarray:
        """Фаза 1: Отримання статистичного орієнтиру (ETS модель)"""
        try:
            # Використовуємо спрощену ETS модель як бейзлайн
            model = ETSModel(series.astype(float), error="add", trend="add", seasonal="add", seasonal_periods=12)
            fit = model.fit(disp=False)
            # Прогноз на період навчання + горизонт
            full_pred = fit.predict(start=0, end=len(series) + self.cfg.horizon - 1)
            return full_pred.values
        except:
            # Fallback: просто середнє значення
            avg = series.mean()
            return np.full(len(series) + self.cfg.horizon, avg)

    def process_single_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Основний пайплайн для однієї групи (напр. одного магазину)"""
        group_df = group_df.sort_values(self.cfg.date_idx).reset_index(drop=True)
        
        if len(group_df) < self.cfg.min_obs:
            return pd.DataFrame()

        # 1. Бейзлайн
        history = group_df[self.cfg.target].dropna()
        baseline_values = self._get_statistical_baseline(history)
        
        # Створюємо майбутні дати
        last_date = pd.to_datetime(group_df[self.cfg.date_idx].max())
        future_dates = pd.date_range(last_date, periods=self.cfg.horizon + 1, freq=self.cfg.frequency)[1:]
        
        future_df = pd.DataFrame({self.cfg.date_idx: future_dates})
        for g in self.cfg.groups:
            future_df[g] = group_df[g].iloc[0]
            
        full_df = pd.concat([group_df, future_df], ignore_index=True)
        full_df['baseline'] = baseline_values[:len(full_df)]

        # 2. Ознаки
        full_df = self._build_temporal_features(full_df)
        full_df = self._build_lag_features(full_df)
        
        # Визначаємо список колонок для навчання (виключаючи цільову та дату)
        exclude = [self.cfg.target, self.cfg.date_idx] + self.cfg.groups
        self.features = [c for c in full_df.columns if c not in exclude]

        # 3. Навчання ML моделі (XGBoost)
        train_data = full_df[full_df[self.cfg.target].notna()].copy()
        test_data = full_df[full_df[self.cfg.target].isna()].copy()
        
        if train_data.empty or test_data.empty:
            return pd.DataFrame()

        X = train_data[self.features].fillna(0)
        y = train_data[self.cfg.target]

        # Спрощене навчання без складного пошуку параметрів для швидкості
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=self.cfg.seed,
            objective='reg:squarederror'
        )
        
        model.fit(X, y)
        
        # Прогноз
        test_data['prediction'] = model.predict(test_data[self.features].fillna(0))
        
        # Повертаємо лише результат
        return test_data[[self.cfg.date_idx] + self.cfg.groups + ['baseline', 'prediction']]

    def run(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Запуск прогнозу для всієї мережі"""
        all_results = []
        
        # Групування за магазинами/категоріями
        grouped = raw_df.groupby(self.cfg.groups)
        
        for name, group in grouped:
            logger.info(f"Обробка групи: {name}")
            res = self.process_single_group(group)
            if not res.empty:
                all_results.append(res)
        
        if not all_results:
            return pd.DataFrame()
            
        return pd.concat(all_results, ignore_index=True)

# Приклад ініціалізації:
# settings = ModelSettings(date_idx='sales_date', target='turnover_amt', groups=['store_id'], horizon=18)
# predictor = TurnoverPredictor(settings)
# final_forecast = predictor.run(your_dataframe)
