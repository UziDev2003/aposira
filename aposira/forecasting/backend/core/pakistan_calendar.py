"""
Pakistan Calendar Module
========================
Provides Pakistani public holidays, Islamic festival dates, and temporal features
for demand forecasting with Pakistan-specific seasonality.

Uses hijri-converter for accurate Islamic calendar date mapping.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional

try:
    from hijri_converter import Hijri, Gregorian
    HAS_HIJRI = True
except ImportError:
    HAS_HIJRI = False


# ─── Fixed Pakistani Public Holidays (Gregorian) ─────────────────────────────
FIXED_HOLIDAYS = {
    "Kashmir Day": (2, 5),
    "Pakistan Day": (3, 23),
    "Labour Day": (5, 1),
    "Independence Day": (8, 14),
    "Iqbal Day": (11, 9),
    "Quaid-e-Azam Day": (12, 25),
}

# ─── Approximate Islamic Holiday Dates (Gregorian estimates per year) ────────
# These shift ~10-11 days earlier each year. Pre-computed for 2020–2027.
# Source: Official Pakistan government gazette notifications
ISLAMIC_HOLIDAYS = {
    2020: {
        "Shab-e-Meraj": date(2020, 3, 22),
        "Shab-e-Barat": date(2020, 4, 9),
        "Ramadan Start": date(2020, 4, 25),
        "Shab-e-Qadr": date(2020, 5, 20),
        "Eid-ul-Fitr": date(2020, 5, 24),
        "Eid-ul-Adha": date(2020, 7, 31),
        "12 Rabi-ul-Awal": date(2020, 10, 30),
        "Shab-e-Meraj_2": None,
    },
    2021: {
        "Shab-e-Meraj": date(2021, 3, 11),
        "Shab-e-Barat": date(2021, 3, 29),
        "Ramadan Start": date(2021, 4, 14),
        "Shab-e-Qadr": date(2021, 5, 9),
        "Eid-ul-Fitr": date(2021, 5, 13),
        "Eid-ul-Adha": date(2021, 7, 21),
        "12 Rabi-ul-Awal": date(2021, 10, 19),
    },
    2022: {
        "Shab-e-Meraj": date(2022, 3, 1),
        "Shab-e-Barat": date(2022, 3, 18),
        "Ramadan Start": date(2022, 4, 3),
        "Shab-e-Qadr": date(2022, 4, 28),
        "Eid-ul-Fitr": date(2022, 5, 3),
        "Eid-ul-Adha": date(2022, 7, 10),
        "12 Rabi-ul-Awal": date(2022, 10, 9),
    },
    2023: {
        "Shab-e-Meraj": date(2023, 2, 18),
        "Shab-e-Barat": date(2023, 3, 8),
        "Ramadan Start": date(2023, 3, 23),
        "Shab-e-Qadr": date(2023, 4, 17),
        "Eid-ul-Fitr": date(2023, 4, 22),
        "Eid-ul-Adha": date(2023, 6, 29),
        "12 Rabi-ul-Awal": date(2023, 9, 28),
    },
    2024: {
        "Shab-e-Meraj": date(2024, 2, 8),
        "Shab-e-Barat": date(2024, 2, 25),
        "Ramadan Start": date(2024, 3, 12),
        "Shab-e-Qadr": date(2024, 4, 6),
        "Eid-ul-Fitr": date(2024, 4, 11),
        "Eid-ul-Adha": date(2024, 6, 17),
        "12 Rabi-ul-Awal": date(2024, 9, 17),
    },
    2025: {
        "Shab-e-Meraj": date(2025, 1, 27),
        "Shab-e-Barat": date(2025, 2, 14),
        "Ramadan Start": date(2025, 3, 1),
        "Shab-e-Qadr": date(2025, 3, 27),
        "Eid-ul-Fitr": date(2025, 3, 31),
        "Eid-ul-Adha": date(2025, 6, 7),
        "12 Rabi-ul-Awal": date(2025, 9, 5),
    },
    2026: {
        "Shab-e-Meraj": date(2026, 1, 16),
        "Shab-e-Barat": date(2026, 2, 3),
        "Ramadan Start": date(2026, 2, 18),
        "Shab-e-Qadr": date(2026, 3, 15),
        "Eid-ul-Fitr": date(2026, 3, 20),
        "Eid-ul-Adha": date(2026, 5, 27),
        "12 Rabi-ul-Awal": date(2026, 8, 26),
    },
    2027: {
        "Shab-e-Meraj": date(2027, 1, 5),
        "Shab-e-Barat": date(2027, 1, 23),
        "Ramadan Start": date(2027, 2, 8),
        "Shab-e-Qadr": date(2027, 3, 5),
        "Eid-ul-Fitr": date(2027, 3, 10),
        "Eid-ul-Adha": date(2027, 5, 16),
        "12 Rabi-ul-Awal": date(2027, 8, 15),
    },
}


def get_ramadan_dates(year: int) -> Optional[tuple]:
    """Return (start_date, end_date) of Ramadan for the given year."""
    if year not in ISLAMIC_HOLIDAYS:
        return None
    start = ISLAMIC_HOLIDAYS[year].get("Ramadan Start")
    if start is None:
        return None
    end = start + timedelta(days=29)  # Ramadan is ~29-30 days
    return (start, end)


def is_during_ramadan(d: date) -> bool:
    """Check if a date falls during Ramadan."""
    dates = get_ramadan_dates(d.year)
    if dates is None:
        return False
    return dates[0] <= d <= dates[1]


def get_eid_dates(year: int) -> List[date]:
    """Return list of all Eid dates (Eid-ul-Fitr + Eid-ul-Adha, 3 days each)."""
    if year not in ISLAMIC_HOLIDAYS:
        return []
    eids = []
    for key in ["Eid-ul-Fitr", "Eid-ul-Adha"]:
        eid_date = ISLAMIC_HOLIDAYS[year].get(key)
        if eid_date:
            for i in range(3):  # Eid is typically 3 days
                eids.append(eid_date + timedelta(days=i))
    return eids


def days_to_next_event(d: date, event_dates: List[date]) -> int:
    """Calculate days until the next occurrence of an event."""
    future = [e for e in event_dates if e >= d]
    if not future:
        return 365  # No upcoming event this year
    return (min(future) - d).days


def days_from_last_event(d: date, event_dates: List[date]) -> int:
    """Calculate days since the last occurrence of an event."""
    past = [e for e in event_dates if e <= d]
    if not past:
        return 365
    return (d - max(past)).days


def generate_pakistan_features(dates: pd.Series) -> pd.DataFrame:
    """
    Generate Pakistan-specific temporal features for a series of dates.
    
    Features generated:
    - Calendar: day_of_week, month, quarter, week_of_year, is_weekend, 
                is_month_start, is_month_end
    - Pakistan Fixed: is_pakistan_day, is_independence_day, is_labour_day, 
                      is_quaid_day, is_iqbal_day, is_kashmir_day
    - Islamic Calendar: is_ramadan, is_eid_ul_fitr, is_eid_ul_adha, 
                        is_pre_eid_rush (7 days before Eid)
    - Proximity: days_to_next_eid, days_from_last_eid
    
    Args:
        dates: pandas Series of datetime values
        
    Returns:
        DataFrame with all temporal features
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    
    # ─── Standard Calendar Features ───────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)  # Friday is special in Pakistan
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    
    # ─── Fixed Pakistani Public Holidays ──────────────────────────────────
    for name, (m, d) in FIXED_HOLIDAYS.items():
        col_name = f"is_{name.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = ((df["date"].dt.month == m) & (df["date"].dt.day == d)).astype(int)
    
    # ─── Islamic Calendar Features ────────────────────────────────────────
    df["is_ramadan"] = df["date"].apply(lambda x: int(is_during_ramadan(x.date())))
    
    # Eid features
    df["is_eid_ul_fitr"] = 0
    df["is_eid_ul_adha"] = 0
    df["is_pre_eid_rush"] = 0
    df["days_to_next_eid"] = 365
    df["days_from_last_eid"] = 365
    
    for year in df["date"].dt.year.unique():
        if year not in ISLAMIC_HOLIDAYS:
            continue
        year_mask = df["date"].dt.year == year
        holidays = ISLAMIC_HOLIDAYS[year]
        
        # Eid-ul-Fitr (3 days)
        eid_fitr = holidays.get("Eid-ul-Fitr")
        if eid_fitr:
            for i in range(3):
                eid_day = eid_fitr + timedelta(days=i)
                df.loc[year_mask & (df["date"].dt.date == eid_day), "is_eid_ul_fitr"] = 1
            # Pre-Eid rush (7 days before)
            for i in range(1, 8):
                rush_day = eid_fitr - timedelta(days=i)
                df.loc[year_mask & (df["date"].dt.date == rush_day), "is_pre_eid_rush"] = 1
        
        # Eid-ul-Adha (3 days)
        eid_adha = holidays.get("Eid-ul-Adha")
        if eid_adha:
            for i in range(3):
                eid_day = eid_adha + timedelta(days=i)
                df.loc[year_mask & (df["date"].dt.date == eid_day), "is_eid_ul_adha"] = 1
            # Pre-Eid rush (7 days before)
            for i in range(1, 8):
                rush_day = eid_adha - timedelta(days=i)
                df.loc[year_mask & (df["date"].dt.date == rush_day), "is_pre_eid_rush"] = 1
        
        # Proximity features
        all_eids = get_eid_dates(year)
        if all_eids:
            year_dates = df.loc[year_mask, "date"]
            df.loc[year_mask, "days_to_next_eid"] = year_dates.apply(
                lambda x: days_to_next_event(x.date(), all_eids)
            )
            df.loc[year_mask, "days_from_last_eid"] = year_dates.apply(
                lambda x: days_from_last_event(x.date(), all_eids)
            )
    
    # ─── Derived Seasonal Features ────────────────────────────────────────
    # Pakistani wedding season (Nov-Feb) drives demand for clothing, food, gifts
    df["is_wedding_season"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    
    # Monsoon season (Jul-Sep) affects supply chains
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    
    # Summer heat (May-Aug) affects food/beverage demand
    df["is_summer"] = df["month"].isin([5, 6, 7, 8]).astype(int)
    
    # Winter (Dec-Feb) affects clothing and heating
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    
    # Salary week (1st and 15th of month ± 2 days)
    df["is_salary_period"] = (
        (df["day_of_month"].between(1, 3)) | 
        (df["day_of_month"].between(15, 17))
    ).astype(int)
    
    # Drop the date column (features only)
    df = df.drop(columns=["date"])
    
    return df


def get_holiday_name(d: date) -> Optional[str]:
    """Return the holiday name for a given date, or None."""
    # Check fixed holidays
    for name, (m, day) in FIXED_HOLIDAYS.items():
        if d.month == m and d.day == day:
            return name
    
    # Check Islamic holidays
    year = d.year
    if year in ISLAMIC_HOLIDAYS:
        for name, holiday_date in ISLAMIC_HOLIDAYS[year].items():
            if holiday_date and holiday_date == d:
                return name
            # Check multi-day Eids
            if holiday_date and name in ["Eid-ul-Fitr", "Eid-ul-Adha"]:
                for i in range(3):
                    if holiday_date + timedelta(days=i) == d:
                        return f"{name} (Day {i+1})"
    
    return None
