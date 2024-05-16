from datetime import datetime, timedelta
from calendar import monthrange

def get_t(date):
    """
    Calculate the time fraction of the year that has passed for a given date.

    Parameters:
    date (datetime): The date for which to calculate the time fraction.

    Returns:
    float: The fraction of the year that has passed as of the given date.
    """
    # Define the beginning of the year for the given date
    bgn_of_year = datetime(date.year, 1, 1)
    # Calculate the difference between the given date and the beginning of the year
    delta = date - bgn_of_year
    # Return the fraction of the year that has passed
    return delta.days / 365.0

def get_T(date):
    """
    Calculate the time to maturity for the next five futures contracts.

    Parameters:
    date (datetime): The current date.

    Returns:
    list: A list of the time to maturity (in years) for the next five futures contracts.
    """
    # Define the mapping of month codes to month numbers
    month_codes = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    # Initialize an empty list to store time to maturity values
    contract_T = []

    # Iterate over the current year and the next year
    for year in [date.year, date.year + 1]:
        # Iterate over the month codes in sorted order of their corresponding months
        for code, month in sorted(month_codes.items(), key=lambda x: x[1]):
            if len(contract_T) >= 5:
                break  # Stop if we have already collected five contracts

            # Determine the last trading day of the month (15th day before month end)
            last_day = monthrange(year, month)[1] - 16
            contract_date = datetime(year, month, last_day)

            # Skip contracts that are too close to maturity
            if date > contract_date - timedelta(days=14):
                continue

            # Calculate the time to maturity in years and add to the list
            time_to_maturity = (contract_date - date).days / 365.0
            contract_T.append(time_to_maturity)

    # If fewer than five contracts were found, pad the list with None values
    while len(contract_T) < 5:
        contract_T.append(None)

    return contract_T
