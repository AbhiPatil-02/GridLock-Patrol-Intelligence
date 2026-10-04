from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LOCAL_TIMEZONE = "Asia/Kolkata"
PRIMARY_EVENT_TIMESTAMP = "created_datetime"


@dataclass
class TimestampProfile:
    column_name: str
    total_rows: int
    valid_timestamps: int
    null_percentage: float

    @property
    def has_usable_values(self) -> bool:
        return self.valid_timestamps > 0

    @property
    def valid_ratio(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.valid_timestamps / self.total_rows


def convert_utc_series_to_local_naive(
    series: pd.Series,
    column_name: str,
    *,
    local_timezone: str = LOCAL_TIMEZONE,
    low_valid_ratio: float = 0.10,
) -> tuple[pd.Series, TimestampProfile]:
    """
    Parse UTC timestamps from the raw feed, convert them to Bengaluru local time,
    and then remove timezone info so downstream CSVs keep the existing naive schema.

    The raw dataset stores timestamps with a `+00` UTC offset. If we strip the
    timezone before converting to Asia/Kolkata, every hour/day/month feature is
    computed on the wrong clock and shifts by roughly +5:30 hours.
    """
    parsed_utc = pd.to_datetime(series, errors="coerce", utc=True)
    total_rows = len(series)
    valid_timestamps = int(parsed_utc.notna().sum())
    null_percentage = (1 - (valid_timestamps / total_rows)) * 100 if total_rows else 0.0

    profile = TimestampProfile(
        column_name=column_name,
        total_rows=total_rows,
        valid_timestamps=valid_timestamps,
        null_percentage=null_percentage,
    )

    print(f"{column_name}:")
    print(f"  total rows: {profile.total_rows}")
    print(f"  valid timestamps: {profile.valid_timestamps}")
    print(f"  null percentage: {profile.null_percentage:.2f}%")

    if not profile.has_usable_values:
        print(
            f"  Skipping {column_name}: 100% missing values. "
            "No timezone conversion or temporal features can be created."
        )
        return parsed_utc.dt.tz_localize(None), profile

    if profile.valid_ratio < low_valid_ratio:
        print(
            f"  Warning: only {profile.valid_ratio:.2%} of rows contain usable "
            "timestamps. Downstream temporal analysis may be unstable."
        )

    localized = parsed_utc.dt.tz_convert(local_timezone)
    naive_local = localized.dt.tz_localize(None)

    print(
        f"  Converted from UTC to {local_timezone} before dropping timezone info."
    )
    return naive_local, profile


def parse_local_naive_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Parse timestamps that were already converted to Bengaluru local time earlier
    in the pipeline and serialized without timezone metadata.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    valid_timestamps = int(parsed.notna().sum())
    total_rows = len(series)
    null_percentage = (1 - (valid_timestamps / total_rows)) * 100 if total_rows else 0.0

    print(f"{column_name}:")
    print(f"  total rows: {total_rows}")
    print(f"  valid timestamps: {valid_timestamps}")
    print(f"  null percentage: {null_percentage:.2f}%")
    print(
        "  Parsed as timezone-naive local event time because upstream preprocessing "
        f"already converted UTC to {LOCAL_TIMEZONE}."
    )
    return parsed


def add_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str,
    *,
    include_week: bool,
    already_localized: bool = False,
) -> TimestampProfile:
    """
    Create local event-time features from the primary violation timestamp.
    """
    print(
        f"Using {timestamp_col} as the primary event timestamp for temporal features."
    )

    if already_localized:
        localized_series = pd.to_datetime(df[timestamp_col], errors="coerce")
        total_rows = len(localized_series)
        valid_timestamps = int(localized_series.notna().sum())
        null_percentage = (
            (1 - (valid_timestamps / total_rows)) * 100 if total_rows else 0.0
        )
        profile = TimestampProfile(
            column_name=timestamp_col,
            total_rows=total_rows,
            valid_timestamps=valid_timestamps,
            null_percentage=null_percentage,
        )
        df[timestamp_col] = localized_series
    else:
        localized_series, profile = convert_utc_series_to_local_naive(
            df[timestamp_col],
            timestamp_col,
        )
        df[timestamp_col] = localized_series

    if not profile.has_usable_values:
        df["hour"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["day_of_week"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df["month"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        if include_week:
            df["week"] = pd.Series(pd.NA, index=df.index, dtype="UInt32")
        df["is_weekend"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        print(
            f"Temporal features were not created because {timestamp_col} contains "
            "no usable timestamps."
        )
        return profile

    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.day_name()
    df["month"] = df[timestamp_col].dt.month
    if include_week:
        df["week"] = df[timestamp_col].dt.isocalendar().week
    weekend_flag = df[timestamp_col].dt.dayofweek >= 5
    df["is_weekend"] = (
        weekend_flag
        .where(df[timestamp_col].notna(), pd.NA)
        .map({True: 1, False: 0})
        .astype("Int64")
    )

    return profile
