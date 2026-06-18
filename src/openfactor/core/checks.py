def require_columns(frame, columns):
    """Raise a clear error when input data is missing required columns.

    Example:
        frame columns = ["ticker", "close"]
        required = ["ticker", "date"]
        error = missing columns: ["date"]
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
