"""고정 CSV를 이용한 과거 시점별 규칙/모델 비교. 포트폴리오 수익률 백테스트가 아니다."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import main


def baseline(ind):
    if ind['close'] > ind['sma20'] > ind['sma60'] and ind['macd_hist'] > 0 and 50 <= ind['rsi14'] <= 70:
        return 'BUY'
    if ind['close'] < ind['sma20'] < ind['sma60'] and ind['macd_hist'] < 0 and 30 <= ind['rsi14'] <= 50:
        return 'SELL'
    return 'HOLD'


def evaluate(df, ticker, name, currency, horizon=5, step=5, threshold=1.0,
             classifier=None):
    df = main.validate_prices(df)
    if horizon < 1 or step < 1 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('horizon/step은 양수, threshold는 유한한 0 이상이어야 합니다.')
    if len(df) < 130 + horizon:
        raise ValueError('평가 구간과 미래 관측 기간을 위한 데이터가 부족합니다.')
    rows = []
    for i in range(129, len(df) - horizon, step):
        history = df.iloc[:i+1]  # 특징에는 기준일 이후 데이터를 전달하지 않는다.
        ind = main.indicators(history)
        context = main.build_context(name, ticker, ind, currency)
        rule = baseline(ind)
        model = classifier(context) if classifier else None
        # 다음 일봉 시가부터 horizon번째 일봉 종가까지의 가격 변화.
        future = float((df.Close.iloc[i+horizon] / df.Open.iloc[i+1] - 1) * 100)
        target = 'BUY' if future > threshold else ('SELL' if future < -threshold else 'HOLD')
        rows.append({'as_of':ind['date'], 'entry_date':df.index[i+1].isoformat(),
                     'exit_date':df.index[i+horizon].isoformat(),
                     'forward_price_change_pct':future, 'target':target,
                     'baseline':rule, 'model':model, 'context_sha256':main.fingerprint(context)})
    model_rows = [r for r in rows if r['model'] and r['model'].get('prediction') is not None]
    return {'evaluation_version':1, 'horizon_sessions':horizon, 'step_sessions':step,
            'neutral_threshold_pct':threshold, 'samples':len(rows),
            'baseline_direction_match_rate':sum(r['baseline']==r['target'] for r in rows)/len(rows),
            'model_coverage':len(model_rows)/len(rows) if classifier else None,
            'model_direction_match_rate':sum(r['model']['prediction']==r['target'] for r in model_rows)/len(model_rows) if model_rows else None,
            'limitations':['방향 라벨 일치율이며 포트폴리오 수익률이 아닙니다.',
                          '거래비용·슬리피지·보유 포지션을 반영하지 않습니다.',
                          'step이 horizon보다 작으면 미래 관측 구간이 겹칩니다.',
                          '수정주가의 사후 수정과 모델 학습 데이터의 미래 정보는 통제하지 않습니다.',
                          '모델의 과거 데이터 기억 및 확률 보정은 검증하지 않습니다.'],
            'rows':rows}


def load_prices(path: Path, timezone_name=None):
    df = pd.read_csv(path, index_col=0, float_precision="round_trip")
    companion = path.parent / "result.json"
    if not timezone_name and companion.exists():
        quality = json.loads(companion.read_text(encoding="utf-8")).get("data_quality", {})
        timezone_name = quality.get("timezone")
        if timezone_name and "fallback" in timezone_name:
            timezone_name = "UTC"
    if timezone_name:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(timezone_name)
    else:
        try:
            df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
        except (ValueError, TypeError) as exc:
            raise ValueError("CSV 시간대 해석 실패: --timezone으로 시장 시간대를 지정하세요.") from exc
    return df


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prices',type=Path,required=True,help='main.py가 저장한 prices.csv')
    p.add_argument('--ticker',required=True)
    p.add_argument('--timezone',help='CSV 시장 시간대; 저장된 result.json이 있으면 자동 적용')
    p.add_argument('--name')
    p.add_argument('--currency',required=True)
    p.add_argument('--horizon',type=int,default=5)
    p.add_argument('--step',type=int,default=5)
    p.add_argument('--threshold',type=float,default=1.0)
    p.add_argument('--with-model',action='store_true')
    p.add_argument('--host',default='http://localhost:11434')
    p.add_argument('--model',default='nimble')
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--retries',type=int,choices=range(4),default=1)
    p.add_argument('--output-dir',type=Path,default=Path('runs/evaluations'))
    args=p.parse_args()
    try:
        if not math.isfinite(args.timeout) or args.timeout<=0:
            raise ValueError('--timeout은 유한한 양수여야 합니다.')
        raw=args.prices.read_bytes()
        df=load_prices(args.prices,args.timezone)
        df,quality=main.prepare_prices(df)
        classifier=(lambda context:main.classify(args.host,args.model,context,main.SCHEMA,'action',args.timeout,args.retries)) if args.with_model else None
        report=evaluate(df,args.ticker,args.name or args.ticker,args.currency,
                        args.horizon,args.step,args.threshold,classifier)
        identity=main.model_identity(args.host,args.model,args.timeout,args.retries) if args.with_model else None
        # 실행 설정/코드/스키마와 실제 평가 입력도 함께 저장한다.
        ind=main.indicators(df)
        record=main.make_record(args.ticker,{'name':args.name or args.ticker,'currency':args.currency},
                                df,ind,'',{},vars(args),quality,identity)
        record['input_file_sha256']=hashlib.sha256(raw).hexdigest()
        record['evaluation_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        record['evaluation']=report
        path=main.save_record(args.output_dir,df,record)
        print(json.dumps({'saved_to':str(path),'summary':{k:v for k,v in report.items() if k!='rows'}},
                         ensure_ascii=False,indent=2,allow_nan=False))
    except (OSError,ValueError,RuntimeError) as exc:
        p.exit(1,f'오류: {exc}\n')


if __name__=='__main__':cli()
