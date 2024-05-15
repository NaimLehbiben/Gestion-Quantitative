from datetime import datetime, timedelta
from calendar import monthrange

def get_t(date):
    bgn_of_df = datetime(1988, 2, 1)  
    delta = date - bgn_of_df
    return delta.days / 365.0

def get_T(date):
    month_codes = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    contract_T = []

    for year in [date.year, date.year + 1]:
        for code, month in sorted(month_codes.items(), key=lambda x: x[1]):
            if len(contract_T) >= 5:
                break
            last_day = monthrange(year, month)[1] - 16
            contract_date = datetime(year, month, last_day)

            if date > contract_date - timedelta(days=14):
                continue

            time_to_maturity = (contract_date - date).days / 365.0
            contract_T.append(time_to_maturity)

    while len(contract_T) < 5:
        contract_T.append(None)  

    return contract_T
