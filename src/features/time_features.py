"""
RetailPulse - Temporal and Calendar Feature Engineering
Extracts day-of-week, month, quarter, UK holidays, cyclical encodings, and time trends.
"""

from typing import Union
import pandas as pd
import numpy as np
import holidays
from src.utils.logger import get_logger

logger = get_logger("time_features")

def extract_calendar_features(
    dates: Union[pd.Series, pd.DatetimeIndex], 
    country_holiday: str = "UK"
) -> pd.DataFrame:
    """
    Extracts rich calendar, cyclical, holiday, and trend features from a series of datetimes.
    
    Args:
        dates: pd.Series or pd.DatetimeIndex containing timestamps.
        country_holiday: Country code for holiday detection (default: 'UK').
        
    Returns:
        pd.DataFrame: Engineered calendar features.
    """
    if isinstance(dates, pd.Series):
        dt = pd.to_datetime(dates)
    else:
        dt = pd.Series(pd.to_datetime(dates))

    feats = pd.DataFrame(index=dt.index)
    feats["date"] = dt.dt.normalize()

    # Core Calendar
    feats["day_of_week"] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    feats["day_name"] = dt.dt.day_name()
    feats["is_weekend"] = (feats["day_of_week"] >= 5).astype(int)
    feats["day_of_month"] = dt.dt.day
    feats["day_of_year"] = dt.dt.dayofyear
    feats["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    feats["month"] = dt.dt.month
    feats["month_name"] = dt.dt.month_name()
    feats["quarter"] = dt.dt.quarter
    feats["year"] = dt.dt.year

    # Boundary flags
    feats["is_month_start"] = dt.dt.is_month_start.astype(int)
    feats["is_month_end"] = dt.dt.is_month_end.astype(int)
    feats["is_quarter_start"] = dt.dt.is_quarter_start.astype(int)
    feats["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    # UK Bank Holiday Flag
    years = dt.dt.year.unique().tolist()
    if country_holiday.upper() in ["UK", "GB", "UNITEDKINGDOM"]:
        uk_holidays = holidays.UnitedKingdom(years=years)
        feats["is_uk_holiday"] = feats["date"].apply(lambda d: 1 if d in uk_holidays else 0)
    else:
        feats["is_uk_holiday"] = 0

    # Cyclical Transformations (preserves continuity: Dec -> Jan, Sun -> Mon)
    feats["sin_month"] = np.sin(2 * np.pi * feats["month"] / 12).round(4)
    feats["cos_month"] = np.cos(2 * np.pi * feats["month"] / 12).round(4)
    feats["sin_day_of_week"] = np.sin(2 * np.pi * feats["day_of_week"] / 7).round(4)
    feats["cos_day_of_week"] = np.cos(2 * np.pi * feats["day_of_week"] / 7).round(4)

    # Normalized Linear Trend index (0 to 1 across the dataset range)
    min_date = feats["date"].min()
    feats["trend_day_index"] = (feats["date"] - min_date).dt.days

    return feats
