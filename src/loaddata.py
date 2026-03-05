import os
from typing import Dict, List

import pandas as pd

import src.config as cfg


REQUIRED_COLUMN_ALIASES = {
    "geo": ["geo", "GEO"],
    "TIME_PERIOD": ["TIME_PERIOD", "time_period", "TIME", "time"],
    "OBS_VALUE": ["OBS_VALUE", "obs_value", "value", "VALUE"],
}


def _resolve_dir(path: str) -> str:
    """Return an existing directory, allowing simple case-insensitive fallback."""
    if os.path.isdir(path):
        return path

    parent = os.path.dirname(path) or "."
    target = os.path.basename(path).lower()
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Parent directory not found: {parent}")

    for entry in os.listdir(parent):
        full_path = os.path.join(parent, entry)
        if os.path.isdir(full_path) and entry.lower() == target:
            return full_path

    raise FileNotFoundError(f"Directory not found: {path}")


def _canonicalize_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Rename compatible source columns to expected canonical names."""
    renamed = {}
    existing = set(df.columns)

    for canonical, aliases in REQUIRED_COLUMN_ALIASES.items():
        match = next((name for name in aliases if name in existing), None)
        if match is None:
            raise KeyError(
                f"{source_name} missing required column for {canonical!r}. "
                f"Found columns: {list(df.columns)}"
            )
        renamed[match] = canonical

    return df.rename(columns=renamed)


def _load_variable_frame(file_path: str, country: str, var_name: str) -> pd.DataFrame:
    """Load one variable file and return [TIME_PERIOD, <var_name>] for one country."""
    raw_df = pd.read_csv(file_path)
    df = _canonicalize_columns(raw_df, source_name=file_path)

    geo_series = df["geo"].astype(str).str.strip().str.casefold()
    country_norm = country.strip().casefold()
    df = df[geo_series == country_norm]

    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format=cfg.DATE_FORMAT, errors="coerce")
    if df["TIME_PERIOD"].isna().all():
        # Fallback for variant formats if strict format parsing fails.
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")

    df[var_name] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df[["TIME_PERIOD", var_name]].dropna(subset=["TIME_PERIOD"])
    return df


def load_country_data(indicator: str, countries: List[str]) -> Dict[str, pd.DataFrame]:
    """Load merged feature/target time series for each country within an indicator folder."""
    indicator_base = os.path.join("data", "raw", "indicator", indicator)

    if os.path.isdir(indicator_base):
        features_dir = _resolve_dir(os.path.join(indicator_base, "features"))
        target_dir = _resolve_dir(os.path.join(indicator_base, "target"))
    else:
        features_dir = _resolve_dir(cfg.FEATURES_DIR)
        target_dir = _resolve_dir(cfg.TARGET_DIR)

    country_data_dict = {}

    for country in countries:
        data_dict = {}

        for file_name in os.listdir(features_dir):
            if not file_name.endswith(".csv"):
                continue
            var_name = file_name.replace(".csv", "")
            df = _load_variable_frame(os.path.join(features_dir, file_name), country, var_name)
            data_dict[var_name] = df

        for file_name in os.listdir(target_dir):
            if not file_name.endswith(".csv"):
                continue
            var_name = file_name.replace(".csv", "")
            df = _load_variable_frame(os.path.join(target_dir, file_name), country, var_name)
            data_dict[var_name] = df

        merged = None
        for dataframe in data_dict.values():
            merged = dataframe if merged is None else pd.merge(merged, dataframe, on="TIME_PERIOD", how="outer")

        if merged is not None and not merged.empty:
            merged.sort_values("TIME_PERIOD", inplace=True)
            merged.set_index("TIME_PERIOD", inplace=True)
            merged = merged.interpolate(method="linear").dropna(how="all")
            country_data_dict[country] = merged

    return country_data_dict
