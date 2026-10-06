import unittest
from unittest.mock import patch
import math
import main
import evaluate
import backtest
import diagnose_schema
from test_core import prices


class FollowupTests(unittest.TestCase):
    def test_direction_schema_horizon_threshold(self):
        schema,field=evaluate.evaluation_schema('direction',10,2)
        self.assertEqual(field,'direction')
        self.assertEqual(schema[field]['choices'],['UP','FLAT','DOWN'])
        self.assertIn('10',schema[field]['description'])
        self.assertIn('2',schema[field]['choice_descriptions']['UP'])
        report=evaluate.evaluate(prices(145),'T','T','USD',objective='direction',classifier=lambda c:{'prediction':'FLAT'})
        self.assertEqual(report['model_direction_match_rate'],1)
        self.assertEqual(report['backtests'],{})

    def test_split_purges_crossing_targets(self):
        df=prices(200)
        cutoff=df.index[155].strftime('%Y-%m-%d')
        seen=[]
        report=evaluate.evaluate(df,'T','T','USD',horizon=10,step=5,split_date=cutoff,classifier=lambda c:seen.append(c) or {'prediction':'HOLD'})
        self.assertEqual(len(seen),report['samples'])
        self.assertGreater(report['purged_samples'],0)
        for row in report['rows']:
            if row['partition']=='development':self.assertLess(row['exit_date'][:10],cutoff)
            if row['partition']=='holdout':self.assertGreaterEqual(row['as_of'],cutoff)
            if row['partition']=='purged':self.assertIsNone(row['model'])

    def test_same_sample_metrics_and_constants(self):
        rows=[{'target':'BUY','baseline':'BUY','model':{'prediction':'BUY'}},
              {'target':'SELL','baseline':'SELL','model':{'prediction':None}},
              {'target':'HOLD','baseline':'BUY','model':{'prediction':'BUY'}}]
        result=evaluate.summarize(rows,['BUY','HOLD','SELL'],True)
        self.assertEqual(result['model']['accuracy'],.5)
        self.assertEqual(result['model']['coverage'],2/3)
        self.assertEqual(result['baseline']['accuracy'],2/3)
        self.assertEqual(result['baseline_same_samples']['accuracy'],.5)
        self.assertEqual(result['constant']['BUY']['accuracy'],1/3)
        self.assertEqual(result['model']['classes']['SELL']['recall'],None)
        self.assertEqual(result['model']['classes']['BUY']['precision'],.5)

    def test_direction_boundary_labels(self):
        df=prices(140)
        df.iloc[134,3]=101;df.iloc[134,1]=103
        report=evaluate.evaluate(df,'T','T','USD',step=10,objective='direction')
        # 정확한 ±threshold 경계의 부동소수점 잔차는 중립으로 처리한다.
        self.assertEqual(report['rows'][0]['target'],'FLAT')

    def test_action_cli_records_actual_schema(self):
        import json,tempfile
        from pathlib import Path
        from test_core import response
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'prices.csv';prices(150).to_csv(csv)
            with patch('sys.argv',['evaluate.py','--prices',str(csv),'--ticker','T','--currency','USD','--objective','action','--with-model','--output-dir',str(Path(tmp)/'out')]), patch('main.request_json',side_effect=lambda url,*a,**k:{'models':[{'name':'nimble:latest','digest':'test'}]} if url.endswith('/api/tags') else response()), patch('builtins.print'):
                evaluate.cli()
            result=json.loads(next((Path(tmp)/'out').glob('*/result.json')).read_text())
            self.assertEqual(list(result['schema']),['action'])
            self.assertTrue(result['prompt_sha256'])
            self.assertTrue(result['evaluation']['backtests'])
            self.assertTrue(next((Path(tmp)/'out').glob('*/backtest.py')).exists())

    def test_benchmark_offline_smoke(self):
        import benchmark,json,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);csv=root/'prices.csv';prices(500).to_csv(csv)
            spec={'split_date':'2026-01-01','horizon':5,'step':20,'threshold':1.,'fee_bps':10,'slippage_bps':5,
                  'datasets':[{'ticker':'T','name':'Test','currency':'USD','prices':str(csv),'previously_observed':False}]}
            manifest=root/'manifest.json';manifest.write_text(json.dumps(spec))
            with patch('sys.argv',['benchmark.py','--manifest',str(manifest),'--output-dir',str(root/'out')]), patch('builtins.print'):
                benchmark.cli()
            summary=json.loads(next((root/'out').glob('*/summary.json')).read_text())
            self.assertEqual(len(summary['results']),2)
            self.assertEqual(summary['results'][0]['objective'],'direction')
            self.assertIsNone(summary['model_identity'])

    def test_cash_hold_no_trade(self):
        df=prices(140)
        row={'as_of':df.index[129].strftime('%Y-%m-%d'),'exit_date':df.index[134].isoformat(),'baseline':'HOLD','model':{'prediction':None}}
        for strategy in ['baseline','model']:
            r=backtest.simulate(df,[row],strategy)
            self.assertEqual(r['net_return_pct'],0)
            self.assertEqual(r['executions'],0)

    def test_next_open_final_exit_fees(self):
        df=prices(140);df.iloc[134,3]=120
        row={'as_of':df.index[129].strftime('%Y-%m-%d'),'exit_date':df.index[134].isoformat(),'baseline':'BUY'}
        r=backtest.simulate(df,[row],fee_bps=10,slippage_bps=5)
        expected=120*(1-.0005)*(1-.001)/(100*(1+.0005)*(1+.001))-1
        self.assertAlmostEqual(r['net_return_pct'],100*expected)
        self.assertEqual(r['trades'][0]['date'],df.index[130].strftime('%Y-%m-%d'))
        self.assertTrue(r['trades'][-1]['forced_final_exit'])
        self.assertEqual(r['completed_trades'],1)
        self.assertAlmostEqual(r['net_return_pct'],r['buy_hold_net_return_pct'])

    def test_sell_and_hold_position_drawdown(self):
        df=prices(140);df.iloc[131,3]=80
        rows=[{'as_of':df.index[129].strftime('%Y-%m-%d'),'exit_date':df.index[134].isoformat(),'baseline':'BUY'},
              {'as_of':df.index[131].strftime('%Y-%m-%d'),'exit_date':df.index[136].isoformat(),'baseline':'HOLD'},
              {'as_of':df.index[133].strftime('%Y-%m-%d'),'exit_date':df.index[138].isoformat(),'baseline':'SELL'}]
        r=backtest.simulate(df,rows,fee_bps=0,slippage_bps=0)
        self.assertEqual(r['executions'],2)
        self.assertAlmostEqual(r['max_drawdown_pct'],20)
        self.assertEqual(r['trades'][-1]['date'],df.index[134].strftime('%Y-%m-%d'))

    def test_cost_validation(self):
        for fee in [-1,10000,float('nan')]:
            with self.assertRaises(ValueError):backtest.simulate(prices(),[],fee_bps=fee)

    def test_diagnostic_variants_do_not_mutate_schema(self):
        variants=diagnose_schema.variants()
        self.assertEqual(variants['reordered']['action']['choices'],['SELL','HOLD','BUY'])
        self.assertEqual(main.SCHEMA['action']['choices'],['BUY','HOLD','SELL'])


if __name__=='__main__':unittest.main()
