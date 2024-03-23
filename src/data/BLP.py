import blpapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange

class BLP():
    def __init__(self):
        self.session = blpapi.Session()
        if not self.session.start():
            print("Failed to start session.")
            return
        if not self.session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return
        self.refDataSvc = self.session.getService('//BLP/refdata')
        print('Session successfully started and service opened')

    def closeSession(self):
        self.session.stop()
        print("Session closed")

    def front_month_contracts(self, date):
        month_codes = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}
        front_month_contracts = []
        checked_months = []

        for year in [date.year, date.year + 1]:
            for code, month in sorted(month_codes.items(), key=lambda x: x[1]):
                if len(front_month_contracts) >= 5:
                    break
                if (code, year) in checked_months:
                    continue
                last_day = monthrange(year, month)[1]
                contract_date = datetime(year, month, last_day)

                if date > contract_date - timedelta(days=14):
                    continue

                front_month_contracts.append((code, year))
                checked_months.append((code, year))

        return front_month_contracts

    def bdh(self, strFields, startdate, enddate):
        column_names = [f"Maturity {i}" for i in range(1, 6)]
        panel_data = pd.DataFrame(index=pd.date_range(startdate, enddate), columns=column_names)

        for each_date in pd.date_range(startdate, enddate):
            contracts = self.front_month_contracts(each_date)
            daily_all_px = {}

            for contract in contracts:
                ticker = f"C {contract[0]}{str(contract[1])[2:]} Comdty"
                request = self.refDataSvc.createRequest('HistoricalDataRequest')

                if isinstance(strFields, str):
                    strFields = [strFields]

                for strF in strFields:
                    request.append('fields', strF)
                request.append('securities', ticker)

                request.set('startDate', each_date.strftime('%Y%m%d'))
                request.set('endDate', each_date.strftime('%Y%m%d'))

                self.session.sendRequest(request)

                while True:
                    event = self.session.nextEvent()
                    if event.eventType() in (blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE):
                        for msg in blpapi.event.MessageIterator(event):
                            securityData = msg.getElement('securityData')
                            fieldDataArray = securityData.getElement('fieldData')
                            for fieldData in fieldDataArray.values():
                                date = fieldData.getElementAsDatetime('date')
                                px_last = fieldData.getElementAsFloat('PX_LAST')
                                daily_all_px[f"{contract[0]}{contract[1]}"] = px_last

                    if event.eventType() == blpapi.Event.RESPONSE:
                        break
                    
            prices = [daily_all_px.get(f"{contract[0]}{contract[1]}", np.nan) for contract in contracts]
            prices_dropNaN = pd.Series(prices).fillna(method='bfill').tolist()

            if prices_dropNaN:
                panel_data.loc[each_date, :] = prices_dropNaN

        return panel_data


