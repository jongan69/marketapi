import pandas as pd

def df_to_list(df, column=None):
    """
    Convert a DataFrame to a list.
    If `column` is specified, returns a list of that column's values.
    Otherwise, returns a list of records (list of dicts).
    """
    if column:
        if column in df.columns:
            return df[column].tolist()
        else:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
    else:
        return df.to_dict(orient='records')

def list_to_df(data, columns=None):
    """
    Convert a list to a DataFrame.
    If `columns` is provided, uses it as column names.
    """
    if not data:
        return pd.DataFrame(columns=columns)
    
    if isinstance(data[0], dict):
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(data, columns=columns)

def series_to_list(series):
    """
    Convert a Pandas Series to a Python list.
    """
    return series.tolist()

def list_to_series(data, name=None):
    """
    Convert a Python list to a Pandas Series.
    """
    return pd.Series(data, name=name)

def get_numeric_value(value, default=None):
    """
    Safely extract numeric values from various data types.
    
    Args:
        value: Input data (pandas Series, list, tuple, int, float, string, or None)
        default: Default value to return if no valid numeric value is found
        
    Returns:
        float or None: The first valid numeric value or default if none found
    """
    if value is None:
        return default
        
    if isinstance(value, (int, float)):
        return float(value)
        
    def extract_from_nested(val):
        """Helper function to extract numeric value from nested structures"""
        if isinstance(val, (list, tuple)):
            for item in val:
                result = extract_from_nested(item)
                if result is not None:
                    return result
            return None
        try:
            if pd.isna(val):
                return None
            return float(val)
        except (TypeError, ValueError):
            return None
        
    if isinstance(value, (list, tuple)):
        result = extract_from_nested(value)
        return result if result is not None else default
        
    if isinstance(value, pd.Series):
        # Get the first non-NaN value
        non_nan_values = value.dropna()
        if not non_nan_values.empty:
            try:
                return float(non_nan_values.iloc[0])
            except (TypeError, ValueError):
                return default
        return default
        
    if hasattr(value, 'values'):
        # Handle other objects with values attribute
        values = [v for v in value.values if pd.notna(v)]
        if values:
            try:
                return float(values[0])
            except (TypeError, ValueError):
                return default
        return default
        
    # Try to convert string to float
    try:
        return float(value)
    except (TypeError, ValueError):
        return default 