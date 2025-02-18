import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def Read_csv(file_path):
    df = pd.read_csv(file_path)

    # Try to detect a datetime column
    for col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # Convert to datetime if possible
        if is_datetime64_any_dtype(df[col]):
            df.set_index(col, inplace=True)
            print(f"Set '{col}' as the datetime index.")
            break
            
    return df
