def forecast_vix(vix_value, horizon):
    """
    Convert VIX to variance forecast
    """
    return horizon / 250 * vix_value**2
