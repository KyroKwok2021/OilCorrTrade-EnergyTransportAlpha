import pandas as pd
import json
import sys
sys.path.append('../../../qwok/')
from signal_oscillator import Price_Oscillator_Signal
from visualization import Visualization as Vis


def dbg(*obj):
        '''
        Log the element for debug
        '''
        for o in obj:
            print(o)


class BacktestHandler:

    def __init__(self, start=None, end=None) -> None:
        self.dict_sector_tickers = {
        'Oil':['USO'],
        'Electric Vehicle': ['TSLA', 'NIO', 'LI', 'XPEV'],
        'Cruise': ['CCL', 'RCL', 'NCLH'],
        'Airline': ['DAL', 'LUV', 'UAL', 'AAL', 'ALK', 'CPA'],
        'Transportation': ['XTN', 'ODFL', 'XPO', 'KNX', 'SNDR', 'WERN', 'RXO'],
        'Energy': ['XLE', 'XOM', 'CVX', 'BP', 'PBR', 'E', 'CVE']
        }
        self.start, self.end = start, end
        self.readRawData()
        df_oil = self.getData('USO')
        self.df_oil = df_oil
        self.open_oil = df_oil['Open']
        self.high_oil = df_oil['High']
        self.low_oil = df_oil['Low']
        self.close_oil = df_oil['Close']
        self.calcPosition_fromOil()
        self.calcFeaturedPosition_fromSector()
        self.calcPositions()


    def clf_pn(self, x):
        '''
        Classify the value is positive or negative
        '''
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    

    def readRawData(self) -> pd.DataFrame:
        '''
        Extract and transform the raw data from csv to pd.Dataframe
        '''
        df_raw = pd.read_csv('Input Data/Data.csv')
        df_raw = df_raw.set_index('Time')
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.drop(['Date', 'Adj Close'], axis=1)
        self.df_raw = df_raw


    def getData(self, ticker) -> pd.DataFrame:
        df = self.df_raw.loc[self.df_raw['Ticker']==ticker].drop(['Ticker'], axis=1).loc[self.start:self.end]
        return df

        
    def calcPosition_fromOil(self):
        '''
        Calculate the signals of different strategy base on Oil Price
        '''
        # Today close return -> tmr signal(0/1)
        signals_closeReturn = self.close_oil.pct_change().apply(lambda x: self.clf_pn(x))

        # Today high and low extrema (extend position) -> tmr signal(0/1)
        temp_high_obj = Price_Oscillator_Signal(self.close_oil, self.high_oil)
        signals_highExtrema = temp_high_obj.getSignals_OscillatorExtrema()
        signals_highExtrema = signals_highExtrema.replace(1, 0)
        temp_low_obj = Price_Oscillator_Signal(self.close_oil, self.low_oil)
        signals_lowExtrema = temp_low_obj.getSignals_OscillatorExtrema()
        signals_lowExtrema = signals_lowExtrema.replace(-1, 0)
        signals_extrema = signals_highExtrema + signals_lowExtrema

        # Today close vs SMA(close) -> tmr signal(0/1)
        window_sma = 10
        sma_close = self.close_oil.rolling(window_sma).mean()
        signals_smaClose = (self.close_oil > sma_close).apply(lambda x: 1 if x else -1)

        # Today open vs SMA(open) -> tdy signal(1)
        sma_open = self.open_oil.rolling(window_sma).mean()
        signals_smaOpen = (self.open_oil > sma_open).apply(lambda x: 1 if x else -1)

        # Today open gap -> tdy signal(1)
        gap = self.open_oil - self.close_oil.shift(1)
        signals_gap = gap.apply(lambda x: self.clf_pn(x))
        
        self.signals_closeReturn = signals_closeReturn.rename('Close Price Return')
        self.signals_extrema = signals_extrema.rename('Extrema')
        self.signals_smaClose = signals_smaClose.rename('Close vs SMA(Close)')
        self.signals_smaOpen = signals_smaOpen.rename('Open vs SMA(Open)')
        self.signals_gap = signals_gap.rename('Gap')
        self.df_oil = pd.concat([self.df_oil, 
                                 signals_gap.rename('Gap'),
                                 sma_open.rename(f'SMA{window_sma}(Open)'),
                                 sma_close.rename(f'SMA{window_sma}(Close)')], 
                                 axis=1)
        return 


    def calcFeaturedPosition_fromSector(self):
            
        # Electric Vehicle & Energy
        self.signals_closeReturn_ev = self.signals_closeReturn
        self.signals_extrema_ev = self.signals_extrema
        self.signals_smaClose_ev = self.signals_smaClose
        self.signals_smaOpen_ev = self.signals_smaOpen.shift(-1)
        self.signals_gap_ev = self.signals_gap.shift(-1)
    
        # Transportation
        self.signals_closeReturn_tn = self.signals_closeReturn * -1
        self.signals_extrema_tn = self.signals_extrema * -1
        self.signals_smaClose_tn = self.signals_smaClose * -1
        self.signals_smaOpen_tn = self.signals_smaOpen.shift(-1) * -1
        self.signals_gap_tn = self.signals_gap.shift(-1) * -1

    def getFeaturedPosition_fromSector(self, sector: str) -> dict:
        if sector.lower() in ['electric vehicle', 'energy']:
            return [self.signals_closeReturn_ev, self.signals_extrema_ev, self.signals_smaClose_ev, self.signals_smaOpen_ev, self.signals_gap_ev]
        else:
            return [self.signals_closeReturn_tn, self.signals_extrema_tn, self.signals_smaClose_tn, self.signals_smaOpen_tn, self.signals_gap_tn]


    def calcPositions(self):
        dict_sector_positionts = {}
        dict_sector_positionts['EV'] = {}
        for position_ds in self.getFeaturedPosition_fromSector('energy'):
            dict_sector_positionts['EV'][position_ds.name] = position_ds
        dict_sector_positionts['TN'] = {}
        for position_ds in self.getFeaturedPosition_fromSector('transportation'):
            dict_sector_positionts['TN'][position_ds.name] = position_ds
        self.dict_sector_positionts = dict_sector_positionts


    def getMetrics(self, 
                _open: pd.Series, 
                _close: pd.Series, 
                _position: pd.Series, 
                _method_position: int,
                _ticker: str,
                _sector: str) -> pd.DataFrame: 
        '''
        Backtest and get the performance metrics given the strategy signals
        '''
        # Backtest the signals
        bt_obj = Vis(close=_close,
                    position=_position,
                    open=_open,
                    method_position=_method_position,
                    extend_position=True,
                    asset_name=_ticker)
        
        # Transform the performance metrics
        res = bt_obj.stat
        res['Strategy'] = _position.name
        res['Method Position'] = _method_position
        res['Ticker'] = _ticker
        res['Sector'] = _sector
        return res


    def handleTickersPositions(self, sector, ls_ticker, ls_position):
        list_res = []
        for ticker in ls_ticker:
            df_ticker = self.getData(ticker)

            for position in ls_position:
                for method_position in [0, 1]:
                    
                    # Next loop 
                    if position.name == 'Gap' and method_position == 0:
                        continue
                    if position.name == 'Open vs SMA(Open)' and method_position == 0:
                        continue
                    
                    dict_metrics = self.getMetrics(
                        _close=df_ticker['Close'],
                        _position=position,
                        _open=df_ticker['Open'],
                        _method_position=method_position,
                        _ticker=ticker,
                        _sector=sector
                        )
                    list_res.append(dict_metrics)
        return list_res


    def run(self):

        list_dict = []
        for sector, tickers in self.dict_sector_tickers.items():

            # Feature the position from difference sectors
            ls_position = self.getFeaturedPosition_fromSector(sector)
            
            # Backtest and get performance metrics for each tickers and position
            this_result = self.handleTickersPositions(sector, tickers, ls_position)

            # Store the result
            list_dict += this_result

        df_result = pd.DataFrame(list_dict)
        return df_result


if __name__ == '__main__':
    Handler = BacktestHandler('2021-12-08', '2023-12-08')
    df_result = Handler.run()

    if 1 == 0:
        df_result.to_excel('Result.xlsx')
        