"""같은 시점에서 전체/단일/순서 변경 스키마를 비교한다. 성능 평가용 데이터가 아니다."""
import argparse
import copy
import json
from pathlib import Path
import main
import evaluate


def variants():
    single={'action':copy.deepcopy(main.SCHEMA['action'])}
    reordered=copy.deepcopy(single)
    reordered['action']['choices']=['SELL','HOLD','BUY']
    return {'full':main.SCHEMA,'action_only':single,'reordered':reordered}


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--samples',type=int,default=6)
    p.add_argument('--output-dir',type=Path,default=Path('runs/diagnostics'))
    a=p.parse_args()
    record=json.loads(a.evaluation.read_text());settings=record['settings']
    df=evaluate.load_prices(Path(settings['prices']))
    rows=record['evaluation']['rows']
    selected=([r for r in rows if r['model']['generated_code']=='D'][:max(1,a.samples-2)] +
              [r for r in rows if r['model']['prediction'] is not None][:2])[:a.samples]
    results=[]
    identity=main.model_identity(settings['host'],settings['model'],60,1)
    for row in selected:
        history=df.loc[df.index.strftime('%Y-%m-%d')<=row['as_of']]
        context=main.build_context(settings['name'],settings['ticker'],main.indicators(history),settings['currency'])
        for label,schema in variants().items():
            body={'model':settings['model'],'prompt':main.build_prompt(context,schema,'action'),
                  'raw':True,'stream':False,'logprobs':True,'top_logprobs':20,
                  'options':{'temperature':0,'num_predict':1}}
            raw=main.request_json(settings['host'].rstrip('/')+'/api/generate',body,60,1)
            parsed=main.parse_classification(raw,schema,'action')
            results.append({'as_of':row['as_of'],'variant':label,'schema':schema,
                            'prompt_sha256':main.fingerprint(body['prompt']),'result':parsed,'raw_response':raw})
            print(row['as_of'],label,repr(parsed['generated_code']),parsed['prediction'],flush=True)
    out={'run_id':main.uuid.uuid4().hex,'created_at':main.datetime.now(main.timezone.utc).isoformat(),
         'model_identity':identity,'identity_after':main.model_identity(settings['host'],settings['model'],60,1),
         'source_evaluation':str(a.evaluation),'rows':results,
         'limitations':['선택된 과거 실패 사례 진단이며 별도 검증 성능이 아님.','서버 모델의 학습 계약과 변환 원본 revision은 digest만으로 확인 불가.']}
    directory=a.output_dir/out['run_id'];directory.mkdir(parents=True)
    (directory/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    (directory/'diagnose_schema.py').write_bytes(Path(__file__).read_bytes())
    print('saved_to',directory)


if __name__=='__main__':cli()
