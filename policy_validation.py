"""사전 고정한 주기·보유 상태·포지션 정책 비교. 최적 정책을 자동 선택하지 않는다."""
import argparse
import json
from pathlib import Path
import main
import evaluate
import portfolio


def action_schema(horizon):
    return {'action':{'type':'enum',
        'description':f'Choose a long/cash position action now. Execute at the next session open and reconsider after {horizon} sessions. Use only supplied facts; future returns are unknown.',
        'choices':['BUY','HOLD','SELL'],'choice_descriptions':{
            'BUY':'If in cash, open the allowed stock allocation; if already holding, retain the position.',
            'HOLD':'Retain the CURRENT position: stay in cash if not holding, retain stock if holding.',
            'SELL':'Close the stock position and retain cash; if already in cash, remain in cash.'}}}


def run_policy(df, ticker, name, currency, horizon=5, step=5, aware=False, cap=1., predictor=None,
               fee_bps=10., slippage_bps=5.):
    df=main.validate_prices(df)
    if step!=horizon or step<1 or len(df)<130+horizon:
        raise ValueError('step=horizon 양수 및 최소 130+horizon개 일봉이 필요합니다.')
    dates=list(df.index.strftime('%Y-%m-%d'));end=len(df)-1
    decisions=list(range(129,len(df)-horizon,step))
    end=decisions[-1]+horizon
    account=portfolio.Portfolio(cap,fee_bps,slippage_bps)
    schema=action_schema(horizon);pending=None;rows=[];curve=[];peak=1.;dd=0.
    for i in range(129,end+1):
        if pending is not None:
            account.execute(pending,float(df.Open.iloc[i]),dates[i]);pending=None
        if i in decisions:
            ind=main.indicators(df.iloc[:i+1]);state=account.state(float(df.Close.iloc[i]))
            context=main.build_context(name,ticker,ind,currency)
            context+=f'\nExecution policy: next session open; reconsider every {horizon} sessions; initial BUY allocation {cap:.0%}; no shorting or leverage.'
            if aware:context+='\nCurrent portfolio at this close: '+main.safe_json(state)
            result=predictor(context,schema) if predictor else {'prediction':evaluate.baseline(ind),'status':'rule'}
            pending=result['prediction']
            rows.append({'as_of':dates[i],'state_at_decision':state,'context':context,
                         'context_sha256':main.fingerprint(context),'schema_sha256':main.fingerprint(schema),'model':result})
        if i>=130:
            if i==end:account.execute('SELL',float(df.Close.iloc[i]),dates[i],True)
            equity=account.equity(float(df.Close.iloc[i]));peak=max(peak,equity);dd=min(dd,equity/peak-1)
            curve.append({'date':dates[i],'equity':equity,'stock_weight':account.state(float(df.Close.iloc[i]))['stock_weight']})
    # 같은 모델 응답의 비용 제거 리플레이: 다른 상태로 모델을 다시 추론하지 않는다.
    replay=portfolio.Portfolio(cap,0,0);signals={r['as_of']:r['model']['prediction'] for r in rows};pending=None
    for i in range(129,end+1):
        if pending is not None:replay.execute(pending,float(df.Open.iloc[i]),dates[i]);pending=None
        if dates[i] in signals:pending=signals[dates[i]]
        if i==end:replay.execute('SELL',float(df.Close.iloc[i]),dates[i],True)
    gross=(replay.cash-1)*100
    bh=portfolio.Portfolio(1,fee_bps,slippage_bps);bh.execute('BUY',float(df.Open.iloc[130]),dates[130])
    bhpeak=1.;bhdd=0.
    for i in range(130,end+1):
        if i==end:bh.execute('SELL',float(df.Close.iloc[i]),dates[i],True)
        eq=bh.equity(float(df.Close.iloc[i]));bhpeak=max(bhpeak,eq);bhdd=min(bhdd,eq/bhpeak-1)
    return {'aware':aware,'cap':cap,'horizon':horizon,'step':step,'schema':schema,
            'start_date':dates[130],'end_date':dates[end],'samples':len(rows),
            'coverage':sum(r['model']['prediction'] is not None for r in rows)/len(rows),
            'net_return_pct':(account.cash-1)*100,'gross_replay_return_pct':gross,
            'cost_drag_pp':gross-(account.cash-1)*100,'max_drawdown_pct':-dd*100,
            'mean_stock_weight':sum(r['stock_weight'] for r in curve)/len(curve),
            'buy_hold_net_return_pct':(bh.cash-1)*100,'buy_hold_max_drawdown_pct':-bhdd*100,
            'trades':account.trades,'equity_curve':curve,'rows':rows,
            'limitations':['비용 제거는 고정 신호 리플레이이며 비용이 달라졌을 때의 새로운 상태 기반 판단이 아님.',
                          'BUY는 초기 cap만 적용하며 이후 가격 변화로 비중이 변할 수 있음.',
                          '수정주가·가정 비용·분수 주식; 세금/현금 이자/장중 낙폭 미반영.']}


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('policy_validation_manifest.json'))
    p.add_argument('--with-model',action='store_true')
    p.add_argument('--host',default='http://localhost:11434');p.add_argument('--model',default='nimble')
    p.add_argument('--output-dir',type=Path,default=Path('runs/policy_validation'))
    a=p.parse_args();spec=json.loads(a.manifest.read_text());root=a.output_dir/main.uuid.uuid4().hex;root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    for file in ['main.py','portfolio.py','policy_validation.py']:(root/file).write_bytes(Path(__file__).with_name(file).read_bytes())
    identity=main.model_identity(a.host,a.model,60,1) if a.with_model else None
    results=[]
    for dataset in spec['datasets']:
        df=evaluate.load_prices(Path(dataset['prices'])) if 'prices' in dataset else main.fetch_prices(dataset['ticker'],dataset.get('period','2y'))
        df,quality=main.prepare_prices(df)
        if spec.get('end_date'):df=df.loc[df.index.strftime('%Y-%m-%d')<=spec['end_date']]
        df=df.iloc[-(130+spec['window_sessions']):]
        (root/(dataset['ticker']+'.csv')).write_text(df.to_csv(float_format='%.17g'))
        for arm in spec['arms']:
            def predict(context,schema):
                r=main.classify(a.host,a.model,context,schema,'action',60,1)
                print(dataset['ticker'],arm['name'],r['prediction'],flush=True);return r
            result=run_policy(df,dataset['ticker'],dataset['name'],dataset['currency'],spec['horizon'],spec['step'],arm['aware'],arm['cap'],predict if a.with_model else None,spec['fee_bps'],spec['slippage_bps'])
            result.update(ticker=dataset['ticker'],arm=arm['name'],data_quality=quality,previously_observed=dataset['previously_observed'],model_identity=identity,snapshot_sha256=main.hashlib.sha256(df.to_csv(float_format='%.17g').encode()).hexdigest())
            results.append(result)
            (root/'results.json').write_text(json.dumps({'manifest':spec,'results':results},ensure_ascii=False,indent=2,allow_nan=False))
            print('completed',dataset['ticker'],arm['name'],result['net_return_pct'],flush=True)
    print('saved_to',root,flush=True)


if __name__=='__main__':cli()
