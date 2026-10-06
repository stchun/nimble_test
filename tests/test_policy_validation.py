import unittest
import json
import pandas as pd
from test_core import prices
import portfolio
import policy_validation as pv


class PolicyTests(unittest.TestCase):
    def test_half_position_fees(self):
        p=portfolio.Portfolio(.5,0,0);p.execute('BUY',100,'day1')
        self.assertEqual(p.cash,.5);self.assertEqual(p.shares,.005)
        p.execute('SELL',120,'day2');self.assertAlmostEqual(p.cash,1.1)

    def test_state_hold_and_repeated_buy(self):
        p=portfolio.Portfolio(.5,0,0);p.execute('BUY',100,'d')
        p.execute('HOLD',110,'e');p.execute('BUY',110,'f')
        self.assertEqual(len(p.trades),1);self.assertAlmostEqual(p.state(110)['unrealized_pct'],10)
        p.execute(None,110,'g');self.assertTrue(p.state(110)['holding'])

    def test_causal_state_and_future_price_invariance(self):
        df=prices(170);contexts=[]
        def predict(c,s):contexts.append(c);return {'prediction':'BUY'}
        first=pv.run_policy(df,'T','T','USD',aware=True,predictor=predict,fee_bps=0,slippage_bps=0)
        second_df=df.copy();second_df.iloc[140:,0:4]*=2
        other=[]
        pv.run_policy(second_df,'T','T','USD',aware=True,predictor=lambda c,s:other.append(c) or {'prediction':'BUY'},fee_bps=0,slippage_bps=0)
        self.assertEqual(contexts[0],other[0])
        self.assertEqual(contexts[1],other[1])
        self.assertFalse(first['rows'][0]['state_at_decision']['holding'])
        self.assertTrue(first['rows'][1]['state_at_decision']['holding'])
        self.assertEqual(first['trades'][0]['date'],df.index[130].strftime('%Y-%m-%d'))
        self.assertEqual(first['samples'],8)

    def test_cash_no_trade_and_cost_replay(self):
        df=prices(170)
        cash=pv.run_policy(df,'T','T','USD',predictor=lambda c,s:{'prediction':None})
        self.assertEqual(cash['net_return_pct'],0)
        self.assertEqual(cash['coverage'],0)
        held=pv.run_policy(df,'T','T','USD',predictor=lambda c,s:{'prediction':'BUY'})
        self.assertEqual(held['gross_replay_return_pct'],0)
        self.assertGreater(held['cost_drag_pp'],0)
        self.assertAlmostEqual(held['net_return_pct'],held['buy_hold_net_return_pct'])

    def test_fixed_signals_half_cap_replay(self):
        import policy_attribution as pa
        df=prices(170);df.iloc[-1,3]=120;df.iloc[-1,1]=122
        original=pv.run_policy(df,'T','T','USD',predictor=lambda c,s:{'prediction':'BUY'},fee_bps=0,slippage_bps=0)
        original['ticker']='T'
        result=pa.replay(df,original,.5,0,0)
        self.assertAlmostEqual(original['net_return_pct'],20)
        self.assertAlmostEqual(result['net_return_pct'],10)
        self.assertTrue(result['inference_reused'])

    def test_attribution_csv_dst_timezone(self):
        import policy_attribution as pa
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        df=prices(170);df.index=df.index.tz_localize('America/New_York')
        arm=pv.run_policy(df,'T','T','USD',aware=True,predictor=lambda c,s:{'prediction':'BUY'})
        arm.update(ticker='T',arm='aware_100',data_quality={'timezone':'America/New_York'})
        spec={'horizon':5,'step':5,'fee_bps':10,'slippage_bps':5,'datasets':[{'ticker':'T','name':'T','currency':'USD'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);df.to_csv(root/'T.csv')
            (root/'results.json').write_text(json.dumps({'manifest':spec,'results':[arm]}))
            with patch('sys.argv',['policy_attribution.py','--run-dir',str(root)]),patch('builtins.print'):
                pa.cli()
            result=json.loads((root/'attribution.json').read_text())
            self.assertEqual(len(result['results']),2)
            self.assertEqual(result['results'][1]['same_signals_cap_50']['samples'],8)

    def test_mismatched_schedule_rejected(self):
        with self.assertRaises(ValueError):pv.run_policy(prices(170),'T','T','USD',step=20)


if __name__=='__main__':unittest.main()
