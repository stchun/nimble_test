"""수정 가격·분수 주식 기준 long/cash 계좌. 미래 가격을 참조하지 않는다."""
import math


class Portfolio:
    def __init__(self, cap=1., fee_bps=10., slippage_bps=5.):
        if not math.isfinite(cap) or not 0 < cap <= 1:
            raise ValueError('cap은 0 초과 1 이하여야 합니다.')
        if any(not math.isfinite(v) or not 0 <= v < 10000 for v in (fee_bps,slippage_bps)):
            raise ValueError('비용은 0~10000 미만의 유한한 bps여야 합니다.')
        self.cap=cap;self.fee=fee_bps/10000;self.slip=slippage_bps/10000
        self.cash=1.;self.shares=0.;self.entry=None;self.trades=[]

    def equity(self, price):return self.cash+self.shares*price

    def state(self, close):
        equity=self.equity(close)
        return {'holding':self.shares>0,'stock_weight':self.shares*close/equity,
                'cash_weight':self.cash/equity,'entry_adjusted_price':self.entry,
                'unrealized_pct':(close/self.entry-1)*100 if self.entry else None}

    def execute(self, signal, price, date, forced=False):
        # BUY는 최초 cap 진입만 수행. HOLD/보류 및 반복 BUY는 비중을 재조정하지 않는다.
        if signal=='BUY' and self.shares==0:
            fill=price*(1+self.slip);budget=self.cash*self.cap
            self.shares=budget/(fill*(1+self.fee));self.cash-=budget;self.entry=fill
            self.trades.append({'date':date,'side':'BUY','price':fill,'shares':self.shares})
        elif signal=='SELL' and self.shares>0:
            fill=price*(1-self.slip);self.cash+=self.shares*fill*(1-self.fee)
            self.trades.append({'date':date,'side':'SELL','price':fill,'shares':self.shares,'forced_final_exit':forced})
            self.shares=0.;self.entry=None
