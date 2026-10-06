import json
import math
import unittest
from unittest.mock import patch
import pandas as pd
import main


def prices(n=130):
    return pd.DataFrame({'Open':100.,'High':102.,'Low':98.,'Close':100.,'Volume':0.},
                        index=pd.date_range('2025-01-01', periods=n))


def response(code='A', entries=None):
    return {'response':code,'logprobs':[{'top_logprobs':entries if entries is not None else
            [{'token':c,'logprob':math.log(p)} for c,p in zip('ABC',[.6,.2,.1])]}]}


class CoreTests(unittest.TestCase):
    def test_invalid_code_abstains(self):
        self.assertIsNone(main.parse_classification(response('X'),main.SCHEMA,'action')['prediction'])

    def test_missing_candidates_abstain(self):
        r=main.parse_classification(response(entries=[{'token':'A','logprob':-8}]),main.SCHEMA,'action')
        self.assertIn('missing_candidates',r['reasons'])
        self.assertIn('invalid_candidate_mass',r['reasons'])

    def test_valid_distribution(self):
        r=main.parse_classification(response(),main.SCHEMA,'action')
        self.assertEqual(r['prediction'],'BUY')
        self.assertAlmostEqual(r['candidate_mass'],.9)
        self.assertAlmostEqual(r['probabilities']['BUY'],2/3)

    def test_malformed_responses(self):
        for data in [{},{'response':'A','logprobs':[]},response(entries=[{'token':'A','logprob':float('nan')}])]:
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                main.parse_classification(data,main.SCHEMA,'action')

    def test_low_mass_and_inconsistent_distribution(self):
        for entries in [
            [{'token':c,'logprob':math.log(.1)} for c in 'ABC'],
            [{'token':c,'logprob':math.log(.3)} for c in 'ABC']+[{'token':'X','logprob':0}],
        ]:
            r=main.parse_classification(response(entries=entries),main.SCHEMA,'action')
            self.assertIsNone(r['prediction'])
            self.assertIn('invalid_candidate_mass',r['reasons'])

    def test_schema_validation_before_request(self):
        with self.assertRaises(ValueError):
            main.build_prompt('context',{'action':{'description':'x','choices':['x']*27}},'action')

    def test_flat_zero_volume_strict_json(self):
        i=main.indicators(prices())
        self.assertEqual(i['rsi14'],50)
        self.assertEqual(i['bb_pct_b'],.5)
        self.assertIsNone(i['volume_ratio_20d'])
        json.dumps(i,default=float,allow_nan=False)
        self.assertNotIn('nan',main.build_context('test','TEST',i))

    def test_range_uses_high_low(self):
        i=main.indicators(prices())
        self.assertEqual((i['high_52w'],i['low_52w']),(102,98))
        self.assertFalse(i['range_complete'])
        self.assertTrue(main.indicators(prices(400))['range_complete'])

    def test_bad_price_data(self):
        for kind in ['nan','duplicate','unsorted','missing','short','negative','bad_ohlc']:
            x=prices()
            if kind=='nan': x.iloc[-1,0]=float('nan')
            if kind=='duplicate': x.index=pd.DatetimeIndex([x.index[0]]*len(x))
            if kind=='unsorted': x=x.iloc[::-1]
            if kind=='missing': x=x.drop(columns='Close')
            if kind=='short': x=x.iloc[:20]
            if kind=='negative': x.iloc[-1,4]=-1
            if kind=='bad_ohlc': x.iloc[-1,1]=90
            with self.subTest(kind=kind), self.assertRaises(ValueError): main.indicators(x)

    def test_wilder_and_bollinger_conventions(self):
        x=prices();x['Close']=[100+i%7 for i in range(130)]
        x['Open']=x.Close;x['High']=x.Close+2;x['Low']=x.Close-2
        i=main.indicators(x)
        delta=x.Close.diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
        ag,al=g.iloc[1:15].mean(),l.iloc[1:15].mean()
        for a,b in zip(g.iloc[15:],l.iloc[15:]):ag,al=(13*ag+a)/14,(13*al+b)/14
        self.assertAlmostEqual(i['rsi14'],100-100/(1+ag/al))
        self.assertAlmostEqual(i['bb_upper'],x.Close.iloc[-20:].mean()+2*x.Close.iloc[-20:].std(ddof=0))


if __name__=='__main__': unittest.main()
