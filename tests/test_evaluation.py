import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import evaluate
import main
from test_core import prices,response


class EvaluationTests(unittest.TestCase):
    def test_no_future_context_leak(self):
        x=prices(145)
        contexts=[]
        def classify(c):
            contexts.append(c)
            return {'prediction':'HOLD'}
        a=evaluate.evaluate(x,'TEST','Test','USD',horizon=5,step=5,classifier=classify)
        modified=x.copy();modified.iloc[130:,0:4]*=2
        other=[]
        evaluate.evaluate(modified,'TEST','Test','USD',horizon=5,step=5,
                          classifier=lambda c:other.append(c) or {'prediction':'HOLD'})
        self.assertEqual(contexts[0],other[0])
        self.assertEqual(a['samples'],3)
        self.assertEqual(a['model_coverage'],1)
        self.assertEqual(a['model_direction_match_rate'],1)

    def test_next_open_entry_and_horizon_close(self):
        x=prices(136)
        x.iloc[130]=[110,112,98,100,0]
        x.iloc[134]=[121,123,119,121,0]
        r=evaluate.evaluate(x,'TEST','Test','USD',horizon=5)['rows'][0]
        self.assertAlmostEqual(r['forward_price_change_pct'],10)
        self.assertEqual(r['target'],'BUY')

    def test_abstention_and_invalid_parameters(self):
        x=prices(140)
        r=evaluate.evaluate(x,'TEST','Test','USD',classifier=lambda c:{'prediction':None})
        self.assertEqual(r['model_coverage'],0)
        self.assertIsNone(r['model_direction_match_rate'])
        for kwargs in [{'horizon':0},{'step':0},{'threshold':float('nan')}]:
            with self.assertRaises(ValueError):evaluate.evaluate(x,'T','T','USD',**kwargs)

    def test_record_snapshot_and_unique_directories(self):
        x=prices()
        record=main.make_record('TEST',{},x,main.indicators(x),'',{}, {},{})
        with tempfile.TemporaryDirectory() as tmp:
            out=main.save_record(Path(tmp),x,record)
            saved=json.loads((out/'result.json').read_text())
            self.assertEqual(saved['snapshot_sha256'],hashlib.sha256((out/'prices.csv').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):main.save_record(Path(tmp),x,record)

    def test_csv_timezone_keeps_market_date(self):
        x=prices();x.index=x.index.tz_localize('Asia/Seoul')
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'prices.csv';x.to_csv(csv)
            loaded=evaluate.load_prices(csv)
            self.assertEqual(loaded.index[-1].date(),x.index[-1].date())
            (Path(tmp)/'result.json').write_text(json.dumps({'data_quality':{'timezone':'Asia/Seoul'}}))
            loaded=evaluate.load_prices(csv)
            self.assertEqual(str(loaded.index.tz),'Asia/Seoul')
            self.assertEqual(loaded.index[-1],x.index[-1])

    def test_main_json_end_to_end(self):
        x=prices(140)
        with tempfile.TemporaryDirectory() as tmp, patch('sys.argv',['main.py','--name','Test','--currency','USD','--json','--output-dir',tmp]), patch('main.fetch_prices',return_value=x), patch('main.request_json',side_effect=lambda url,*a,**k: {'models':[{'name':'nimble:latest','digest':'sha256:test'}]} if url.endswith('/api/tags') else response()), patch('builtins.print') as output:
            main.main()
            record=json.loads(output.call_args_list[-1].args[0])
            self.assertEqual(record['results']['action']['prediction'],'BUY')
            self.assertTrue(record['model_identity']['verified'])
            self.assertEqual(len(list(Path(tmp).glob('*/prices.csv'))),1)


if __name__=='__main__':unittest.main()
