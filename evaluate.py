"""과거 시점의 방향 분류와 매매 행동을 분리 평가하고 action은 long/cash 백테스트한다."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import main
import backtest


def baseline(ind):
    if ind['close'] > ind['sma20'] > ind['sma60'] and ind['macd_hist'] > 0 and 50 <= ind['rsi14'] <= 70:
        return 'BUY'
    if ind['close'] < ind['sma20'] < ind['sma60'] and ind['macd_hist'] < 0 and 30 <= ind['rsi14'] <= 50:
        return 'SELL'
    return 'HOLD'


def evaluation_schema(objective, horizon=5, threshold=1.0, scope="single"):
    if objective == "action":
        return ({"action": main.SCHEMA["action"]} if scope == "single" else main.SCHEMA), "action"
    field = {"type":"enum", "description":f"Predict price direction from the next session open to session {horizon} close using only the historical indicators.",
             "choices":["UP","FLAT","DOWN"], "choice_descriptions":{
                 "UP":f"Future price change greater than +{threshold}%.",
                 "FLAT":f"Future price change within [-{threshold}%, +{threshold}%].",
                 "DOWN":f"Future price change less than -{threshold}%."}}
    return {"direction":field}, "direction"


def classification_metrics(rows, get_prediction, labels):
    predicted=[(r, get_prediction(r)) for r in rows]
    valid=[(r,p) for r,p in predicted if p in labels]
    matrix={p:{t:sum(v==p and r["target"]==t for r,v in valid) for t in labels} for p in labels}
    classes={}
    for label in labels:
        tp=matrix[label][label]
        n_pred=sum(matrix[label].values())
        n_actual=sum(matrix[p][label] for p in labels)
        precision=tp/n_pred if n_pred else None
        recall=tp/n_actual if n_actual else None
        f1=2*tp/(n_pred+n_actual) if n_pred+n_actual else None
        classes[label]={"precision":precision,"recall":recall,"f1":f1,"predicted":n_pred,"support":n_actual}
    hits=sum(r["target"]==p for r,p in valid)
    return {"samples":len(rows),"predicted_samples":len(valid),"correct":hits,
            "coverage":len(valid)/len(rows) if rows else None,
            "accuracy":hits/len(valid) if valid else None,
            "correct_fraction_all":hits/len(rows) if rows else None,
            "confusion_matrix_predicted_by_actual":matrix,"classes":classes,
            "balanced_accuracy":sum(v["recall"] for v in classes.values() if v["recall"] is not None)/sum(v["recall"] is not None for v in classes.values()) if valid else None}


def summarize(rows, labels, with_model):
    model=lambda r:(r.get("model") or {}).get("prediction")
    same=[r for r in rows if model(r) in labels]
    metrics={"baseline":classification_metrics(rows,lambda r:r["baseline"],labels),
             "constant":{label:classification_metrics(rows,lambda r,l=label:l,labels) for label in labels}}
    if with_model:
        metrics["model"]=classification_metrics(rows,model,labels)
        metrics["baseline_same_samples"]=classification_metrics(same,lambda r:r["baseline"],labels)
        metrics["constant_same_samples"]={label:classification_metrics(same,lambda r,l=label:l,labels) for label in labels}
    return metrics


def evaluate(df, ticker, name, currency, horizon=5, step=5, threshold=1.0,
             classifier=None, objective="action", split_date=None, fee_bps=10., slippage_bps=5.):
    df = main.validate_prices(df)
    if objective not in ("action","direction"):
        raise ValueError("objective는 action/direction이어야 합니다.")
    if horizon < 1 or step < 1 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('horizon/step은 양수, threshold는 유한한 0 이상이어야 합니다.')
    if len(df) < 130 + horizon:
        raise ValueError('평가 구간과 미래 관측 기간을 위한 데이터가 부족합니다.')
    if split_date:
        split_date=pd.Timestamp(split_date).strftime('%Y-%m-%d')
    labels=["BUY","HOLD","SELL"] if objective=="action" else ["UP","FLAT","DOWN"]
    rows = []
    for i in range(129, len(df) - horizon, step):
        history = df.iloc[:i+1]
        ind = main.indicators(history)
        context = main.build_context(name, ticker, ind, currency)
        rule = baseline(ind)
        if objective=="direction":rule={"BUY":"UP","HOLD":"FLAT","SELL":"DOWN"}[rule]
        future = float((df.Close.iloc[i+horizon] / df.Open.iloc[i+1] - 1) * 100)
        at_boundary = math.isclose(abs(future), threshold, rel_tol=1e-12, abs_tol=1e-10)
        target = labels[0] if future > threshold and not at_boundary else (labels[2] if future < -threshold and not at_boundary else labels[1])
        exit_date=df.index[i+horizon].isoformat()
        partition="all" if not split_date else ("holdout" if ind['date']>=split_date else "development" if exit_date[:10]<split_date else "purged")
        # 분할 경계를 가로지르는 정답은 개발/검증 지표에서 제외한다.
        model = classifier(context) if classifier and partition!="purged" else None
        rows.append({'as_of':ind['date'], 'entry_date':df.index[i+1].isoformat(),
                     'exit_date':exit_date,'partition':partition,
                     'forward_price_change_pct':future, 'target':target,
                     'baseline':rule, 'model':model, 'context_sha256':main.fingerprint(context)})
    with_model=classifier is not None
    usable=[r for r in rows if r['partition']!='purged']
    metrics=summarize(usable,labels,with_model)
    splits={part:summarize([r for r in rows if r['partition']==part],labels,with_model) for part in ['development','holdout']} if split_date else {}
    if split_date and any(not any(r['partition']==part for r in rows) for part in ['development','holdout']):
        raise ValueError('split-date 양쪽에 평가 표본이 필요합니다.')
    if objective=="action":
        partitions={'all':usable} if not split_date else {part:[r for r in rows if r['partition']==part] for part in ['development','holdout']}
        simulations={part:{signal:backtest.simulate(df,rs,signal,fee_bps,slippage_bps) for signal in (['baseline','model'] if with_model else ['baseline'])} for part,rs in partitions.items()}
    else:simulations={}
    return {'evaluation_version':2,'objective':objective,
            'label_semantics':'future_direction' if objective=='direction' else 'legacy_direction_proxy_for_action; HOLD is not no-trade correctness',
            'horizon_sessions':horizon,'step_sessions':step,'neutral_threshold_pct':threshold,
            'samples':len(usable),'purged_samples':len(rows)-len(usable),'split_date':split_date,
            'baseline_direction_match_rate':metrics['baseline']['accuracy'],
            'model_coverage':metrics['model']['coverage'] if with_model else None,
            'model_direction_match_rate':metrics['model']['accuracy'] if with_model else None,
            'metrics':metrics,'splits':splits,'backtests':simulations,
            'limitations':['동일한 가격 라벨의 방향 평가와 매매 행동 백테스트는 별개입니다.',
                          '분할은 시간 순서 기준이며 이미 본 과거 자료를 미관측 데이터로 만들지 않습니다.',
                          'step < horizon이면 미래 관측 구간이 겹칩니다.',
                          '수정주가의 사후 수정과 모델 학습 데이터의 미래 정보/과거 기억은 통제하지 않습니다.'],
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
    p.add_argument('--objective',choices=['direction','action'],default='direction')
    p.add_argument('--schema-scope',choices=['single','full'],default='single')
    p.add_argument('--split-date',help='검증 시작일 YYYY-MM-DD; 경계 교차 표본 제외')
    p.add_argument('--fee-bps',type=float,default=10)
    p.add_argument('--slippage-bps',type=float,default=5)
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
        schema,field=evaluation_schema(args.objective,args.horizon,args.threshold,args.schema_scope)
        def predict(context):
            result=main.classify(args.host,args.model,context,schema,field,args.timeout,args.retries)
            print(f'평가: {field} {result["generated_code"]!r} {result["status"]}',file=sys.stderr,flush=True)
            return result
        classifier=predict if args.with_model else None
        report=evaluate(df,args.ticker,args.name or args.ticker,args.currency,
                        args.horizon,args.step,args.threshold,classifier,args.objective,args.split_date,args.fee_bps,args.slippage_bps)
        identity=main.model_identity(args.host,args.model,args.timeout,args.retries) if args.with_model else None
        # 실행 설정/코드/스키마와 실제 평가 입력도 함께 저장한다.
        ind=main.indicators(df)
        record=main.make_record(args.ticker,{'name':args.name or args.ticker,'currency':args.currency},
                                df,ind,'',{},vars(args),quality,identity)
        record['input_file_sha256']=hashlib.sha256(raw).hexdigest()
        record['evaluation_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        record["schema"]=schema
        record["schema_sha256"]=main.fingerprint(schema)
        record["prompt_sha256"]={row["as_of"]: row["model"].get("prompt_sha256") for row in report["rows"] if row["model"]}
        record["backtest_source_sha256"]=hashlib.sha256(Path(backtest.__file__).read_bytes()).hexdigest()
        record['evaluation_schema']=schema
        record['evaluation_schema_sha256']=main.fingerprint(schema)
        record['evaluation']=report
        path=main.save_record(args.output_dir,df,record)
        print(json.dumps({'saved_to':str(path),'summary':{k:v for k,v in report.items() if k not in ('rows','backtests')}},
                         ensure_ascii=False,indent=2,allow_nan=False))
    except (OSError,ValueError,RuntimeError) as exc:
        p.exit(1,f'오류: {exc}\n')


if __name__=='__main__':cli()
