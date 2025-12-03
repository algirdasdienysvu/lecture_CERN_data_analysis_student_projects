import pandas as pd

def remove_empty_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:

    # print('Removing completely empty rows and columns...')
    # Work on a copy so you don't mutate original by accident
    df_clean = df.copy()

    # 1. Turn pure-whitespace cells into NaN (but keep other values)
    df_clean = df_clean.replace(r'^\s*$', pd.NA, regex=True)

    # 2. Show how many rows/cols are fully empty
    empty_row_mask = df_clean.isna().all(axis=1)
    empty_col_mask = df_clean.isna().all(axis=0)

    print(f"Completely empty rows found: {empty_row_mask.sum()}")
    print(f"Completely empty columns found: {empty_col_mask.sum()}")

    # 3. Drop only fully-empty rows/columns
    df_clean = df_clean.loc[~empty_row_mask, ~empty_col_mask]

    return df_clean

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # print('Stripping whitespace from column names and cell values...')
   
    # Work on a copy for safety
    df_clean = df.copy()

    # 1) Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()

    # 2) Strip whitespace inside cells (only object/string columns)
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean

def normalize_missing_values(df):

    df_clean = df.copy()

# Add more patterns as needed   
    missing_patterns = [
        r"^\s*$",     # empty / whitespace
        r"(?i)^na$",  
        r"(?i)^n/a$",
        r"(?i)^null$",
        r"(?i)^none$",
        r"^\?$",
        r"^-$",
        r"^\.$",
    ]

    for pattern in missing_patterns:
        df_clean = df_clean.replace(pattern, pd.NA, regex=True)

    return df_clean

def fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == "object":  
            # Replace comma-decimal only if the value looks like a number
            df_clean[col] = (
                df_clean[col]
                .str.replace(r"(?<=\d),(?=\d)", ".", regex=True)  # 1,25 -> 1.25
            )

            # Convert to numeric where possible
            df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

    return df_clean

def extract_numeric_and_unit(df):
    
    import re

    df_clean = df.copy()

    # number (with . or ,) + optional space + unit letters
    pattern = re.compile(r"^\s*([0-9]+[.,]?[0-9]*)\s*([A-Za-zµ°%]+)\s*$")

    new_columns_order = []

    for col in df_clean.columns:
        # Only try to split object/string columns
        if df_clean[col].dtype == "object":
            series = df_clean[col].astype(str)

            # Extract number + unit into two columns (0 = number, 1 = unit)
            extracted = series.str.extract(pattern)

            # If this column actually matches the pattern at least once,
            # we treat it as numeric+unit and replace it
            if not extracted.isna().all().all():
                num_col = f"{col}_value"
                unit_col = f"{col}_unit"

                # Convert number: change comma to dot, then to float
                nums = extracted[0].str.replace(",", ".", regex=False)
                df_clean[num_col] = pd.to_numeric(nums, errors="coerce")
                df_clean[unit_col] = extracted[1]

                # Put these two where the original column was
                new_columns_order.extend([num_col, unit_col])
                continue  # skip adding the original column name

        # If not object type or no match: keep original column as-is
        new_columns_order.append(col)

    # Reorder to reflect replacements
    df_clean = df_clean[new_columns_order]

    return df_clean

def convert_units_to_SI(df: pd.DataFrame) -> pd.DataFrame:

    import math

    df_clean = df.copy()

    def _convert_one(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return value, unit

        try:
            v = float(value)
        except (TypeError, ValueError):
            return value, unit

        u = str(unit).strip()
        u = u.replace("°", "deg")  # normalize degrees
        u = u.replace("µ", "u")    # normalize micro
        u = u.lower()

        # ---- temperature -> K ----
        if u in ("degc", "c"):
            return v + 273.15, "K"
        if u in ("degf", "f"):
            return (v - 32.0) * 5.0 / 9.0 + 273.15, "K"
        if u in ("k", "degk"):
            return v, "K"

        # ---- length -> m ----
        length_units = {
            "mm": 1e-3,
            "cm": 1e-2,
            "m": 1.0,
            "km": 1e3,
        }
        if u in length_units:
            return v * length_units[u], "m"

        # ---- mass -> kg ----
        mass_units = {
            "mg": 1e-6,
            "g": 1e-3,
            "kg": 1.0,
            "t": 1e3,
        }
        if u in mass_units:
            return v * mass_units[u], "kg"

        # ---- time -> s ----
        time_units = {
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
        }
        if u in time_units:
            return v * time_units[u], "s"

        # ---- pressure -> Pa ----
        pressure_units = {
            "pa": 1.0,
            "kpa": 1e3,
            "mpa": 1e6,
            "bar": 1e5,
            "mbar": 1e2,
            "atm": 101325.0,
            "psi": 6894.757,
        }
        if u in pressure_units:
            return v * pressure_units[u], "Pa"

        # ---- force -> N ----
        force_units = {
            "n": 1.0,
            "kn": 1e3,
        }
        if u in force_units:
            return v * force_units[u], "N"

        # ---- energy -> J ----
        energy_units = {
            "j": 1.0,
            "kj": 1e3,
        }
        if u in energy_units:
            return v * energy_units[u], "J"

        # ---- percentage -> fraction (0–1) ----
        if u in ("%", "pct"):
            return v / 100.0, "1"   # dimensionless

        # unknown unit -> leave as is
        return value, unit

    # Look for <base>_value + <base>_unit pairs
    cols = list(df_clean.columns)
    for col in cols:
        if col.endswith("_value"):
            base = col[:-6]  # remove "_value"
            unit_col = base + "_unit"
            if unit_col in df_clean.columns:
                for idx, (val, unit) in df_clean[[col, unit_col]].iterrows():
                    new_val, new_unit = _convert_one(val, unit)
                    df_clean.at[idx, col] = new_val
                    df_clean.at[idx, unit_col] = new_unit

                # make sure numeric column is numeric dtype
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean