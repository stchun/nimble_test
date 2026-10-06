"""고정 매니페스트로 여러 종목과 시간 분할을 평가한다. 파라미터 탐색은 하지 않는다."""
import argparse
import json
from pathlib import Path
import main
import evaluate


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('benchmark_manifest.json'))
    p.add_argument('--with-model',action='store_true')
    p.add_argument('--host',default='http://localhost:11434')
    p.add_argument('--model',default='nimble')
    p.add_argument('--output-dir',type=Path,default=Path('runs/benchmarks'))
    a=p.parse_args();spec=json.loads(a.manifest.read_text())
    root=a.output_dir/main.uuid.uuid4().hex;root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    identity=main.model_identity(a.host,a.model,60,1) if a.with_model else None
    summary=[]
    for dataset in spec['datasets']:
        ticker=dataset['ticker'];print('dataset',ticker,flush=True)
        df=evaluate.load_prices(Path(dataset['prices'])) if 'prices' in dataset else main.fetch_prices(ticker,dataset.get('period','2y'))
        df,quality=main.prepare_prices(df)
        for objective in ['direction','action']:
            schema,field=evaluate.evaluation_schema(objective,spec['horizon'],spec['threshold'])
            def predict(context):
                r=main.classify(a.host,a.model,context,schema,field,60,1)
                print(ticker,objective,repr(r['generated_code']),r['status'],flush=True)
                return r
            report=evaluate.evaluate(df,ticker,dataset['name'],dataset['currency'],spec['horizon'],spec['step'],spec['threshold'],predict if a.with_model else None,objective,spec['split_date'],spec['fee_bps'],spec['slippage_bps'])
            ind=main.indicators(df)
            record=main.make_record(ticker,dataset,df,ind,'',{},dict(vars(a),**{k:v for k,v in spec.items() if k!='datasets'},objective=objective),quality,identity)
            record.update(evaluation=report,evaluation_schema=schema,evaluation_schema_sha256=main.fingerprint(schema),
                          benchmark_source_sha256=main.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          evaluation_source_sha256=main.hashlib.sha256(Path(evaluate.__file__).read_bytes()).hexdigest(),
                          backtest_source_sha256=main.hashlib.sha256(Path(evaluate.backtest.__file__).read_bytes()).hexdigest())
            record["schema"]=schema
            record["schema_sha256"]=main.fingerprint(schema)
            record["prompt_sha256"]={row["as_of"]: row["model"].get("prompt_sha256") for row in report["rows"] if row["model"]}
            path=main.save_record(root,df,record)
            item={'ticker':ticker,'objective':objective,'previously_observed':dataset['previously_observed'],
                  'saved_to':str(path),'samples':report['samples'],'purged_samples':report['purged_samples'],
                  'splits':report['splits'],
                  'backtests':{part:{strategy:{k:v for k,v in bt.items() if k not in ['equity_curve','trades']} for strategy,bt in strategies.items()} for part,strategies in report['backtests'].items()}}
            summary.append(item)
            # 중간 완료 결과를 즉시 남긴다.
            (root/'summary.json').write_text(json.dumps({'manifest':spec,'model_identity':identity,'results':summary},ensure_ascii=False,indent=2,allow_nan=False))
            print('saved_to',path,flush=True)
    (root/'benchmark.py').write_bytes(Path(__file__).read_bytes())
    print('benchmark_complete',root,flush=True)


if __name__=='__main__':cli()
