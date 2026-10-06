"""매수 신호로 전액 진입, 매도 신호로 전액 청산하는 long/cash 일별 시뮬레이션."""
import math


def simulate(df, rows, signal='baseline', fee_bps=10.0, slippage_bps=5.0):
    if any(not math.isfinite(v) or not 0 <= v < 10000 for v in [fee_bps,slippage_bps]):
        raise ValueError('비용은 유한한 0~10000 미만 bps이어야 합니다.')
    if not rows:
        return {'status':'no_samples'}
    dates=df.index.strftime('%Y-%m-%d')
    positions={d:i for i,d in enumerate(dates)}
    orders={}
    for row in rows:
        idx=positions[row['as_of']]+1
        value=row['baseline'] if signal=='baseline' else (row.get('model') or {}).get('prediction')
        orders[idx]=value
    start=min(orders)
    end=max(positions[str(row['exit_date'])[:10]] for row in rows)
    cash,shares=1.,0.
    fee,slip=fee_bps/10000,slippage_bps/10000
    trades=[];curve=[];peak=1.;drawdown=0.
    for i in range(start,end+1):
        order=orders.get(i)
        if order=='BUY' and shares==0:
            price=float(df.Open.iloc[i])*(1+slip)
            shares=cash/(price*(1+fee));cash=0.
            trades.append({'date':dates[i],'side':'BUY','price':price,'shares':shares})
        elif order=='SELL' and shares>0:
            price=float(df.Open.iloc[i])*(1-slip)
            cash=shares*price*(1-fee)
            trades.append({'date':dates[i],'side':'SELL','price':price,'shares':shares})
            shares=0.
        if i==end and shares>0:
            price=float(df.Close.iloc[i])*(1-slip)
            cash=shares*price*(1-fee)
            trades.append({'date':dates[i],'side':'SELL','price':price,'shares':shares,'forced_final_exit':True})
            shares=0.
        equity=cash+shares*float(df.Close.iloc[i])
        peak=max(peak,equity);drawdown=min(drawdown,equity/peak-1)
        curve.append({'date':dates[i],'equity':equity})
    buyhold=float(df.Close.iloc[end])*(1-slip)*(1-fee)/(float(df.Open.iloc[start])*(1+slip)*(1+fee))-1
    return {'status':'ok','policy':'long_cash_all_in; HOLD/abstention retain position; next open; final close liquidation',
            'fee_bps_per_side':fee_bps,'slippage_bps_per_side':slippage_bps,
            'start_date':dates[start],'end_date':dates[end],
            'net_return_pct':(curve[-1]['equity']-1)*100,
            'max_drawdown_pct':-drawdown*100,'buy_hold_net_return_pct':buyhold*100,
            'executions':len(trades),'completed_trades':sum(t['side']=='SELL' for t in trades),
            'trades':trades,'equity_curve':curve,
            'limitations':['수정 OHLC 사용; 현금 이자·세금·배당 현금흐름 별도 미반영.',
                          '일별 종가 낙폭이며 장중 낙폭은 미반영.',
                          '수수료와 슬리피지는 가정값이며 실제 체결/유동성을 보장하지 않음.']}
