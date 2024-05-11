from datetime import datetime, timedelta
from calendar import monthrange


def get_t(date):
    end_of_year = datetime(date.year, 12, 31)  
    delta = end_of_year - date 
    return delta.days / 365

def get_T(date):
    month_codes = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
    contract_T = []

    for year in [date.year, date.year + 1]:
        for code, month in sorted(month_codes.items(), key=lambda x: x[1]):
            if len(contract_T) >= 5:
                break
            last_day = monthrange(year, month)[1] -16
            contract_date = datetime(year, month, last_day)

            if date > contract_date - timedelta(days=14):
                continue

            contract_T.append(contract_date)

    if len(contract_T) == 5:
        return contract_T
    else:
        return None