"""저장된 동일 신호를 다른 초기 비중에 리플레이해 실행 정책 영향을 분리한다."""
import argparse
import json
from pathlib import Path
import main
import evaluate
import backtest
import policy_validation as pv


def replay(df, report, cap, fee_bps=10., slippage_bps=5.):
    signals=iter(report['rows'])
    result=pv.run_policy(df,report['ticker'],report['ticker'],'UNKNOWN',report['horizon'],report['step'],
                         report['aware'],cap,lambda c,s:next(signals)['model'],fee_bps,slippage_bps)
    assert [r['as_of'] for r in result['rows']]==[r['as_of'] for r in report['rows']]
    result['inference_reused']=True
    result['replay_note']='모델의 상태 피드백 응답은 재계산하지 않고 같은 신호를 다른 cap에 실행함.'
    return result


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();raw=json.loads((a.run_dir/'results.json').read_text());spec=raw['manifest'];results=[]
    for dataset in spec['datasets']:
        arms=[r for r in raw['results'] if r['ticker']==dataset['ticker']]
        timezone=arms[0]['data_quality']['timezone']
        if 'fallback' in timezone:timezone='UTC'
        df=evaluate.load_prices(a.run_dir/(dataset['ticker']+'.csv'),timezone)
        passive=pv.run_policy(df,dataset['ticker'],dataset['name'],dataset['currency'],spec['horizon'],spec['step'],False,.5,lambda c,s:{'prediction':'BUY'},spec['fee_bps'],spec['slippage_bps'])
        results.append({'ticker':dataset['ticker'],'arm':'passive_50_cash_50','net_return_pct':passive['net_return_pct'],'max_drawdown_pct':passive['max_drawdown_pct'],'mean_stock_weight':passive['mean_stock_weight']})
        for arm in arms:
            # 계좌/프롬프트 설정은 고정된 원래 결과이며 새 추론 없음.
            item={k:v for k,v in arm.items() if k not in ['rows','trades','equity_curve','schema','data_quality']}
            if arm['cap']==1:
                fixed=replay(df,arm,.5,spec['fee_bps'],spec['slippage_bps'])
                item['same_signals_cap_50']={k:v for k,v in fixed.items() if k not in ['rows','trades','equity_curve','schema']}
            results.append(item)
        if dataset.get('reference_result'):
            old=json.loads(Path(dataset['reference_result']).read_text())
            first=df.index[129].strftime('%Y-%m-%d');last=df.index[-6].strftime('%Y-%m-%d')
            rows=[r for r in old['evaluation']['rows'] if first<=r['as_of']<=last and r['partition']=='holdout']
            rows=[{'as_of':first,'exit_date':df.index[-1].isoformat(),'baseline':'HOLD','model':{'prediction':'HOLD'}}]+rows
            rows[-1]=dict(rows[-1],exit_date=df.index[-1].isoformat())
            reference=backtest.simulate(df,rows,'model')
            results.append({'ticker':dataset['ticker'],'arm':'existing_sparse_signals_cash_reset',
                            'net_return_pct':reference['net_return_pct'],'max_drawdown_pct':reference['max_drawdown_pct'],
                            'buy_hold_net_return_pct':reference['buy_hold_net_return_pct'],
                            'note':'같은 시작/종료일, 현금 초기화. 기존 sparse 신호 재사용; 새 스키마와 주기 모두 달라 순수 주기 효과만 뜻하지 않음.'})
    out={'manifest':spec,'results':results,'created_at':main.datetime.now(main.timezone.utc).isoformat()}
    (a.run_dir/'attribution.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':cli()
