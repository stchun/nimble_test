import io
import unittest
from unittest.mock import patch
import urllib.error
import pandas as pd
import main
from test_core import prices


class IOTests(unittest.TestCase):
    def test_current_day_excluded_and_age_warning(self):
        x=prices(131)
        df,q=main.prepare_prices(x,now=x.index[-1])
        self.assertEqual(len(df),130)
        self.assertNotEqual(df.index[-1],x.index[-1])
        _,q=main.prepare_prices(x,now=x.index[-1]+pd.Timedelta(days=10))
        self.assertEqual(q['age_calendar_days'],10)
        self.assertTrue(q['warnings'])

    def test_include_current_and_future_rejected(self):
        x=prices()
        df,q=main.prepare_prices(x,now=x.index[-1],include_current=True)
        self.assertEqual(len(df),130)
        self.assertTrue(any('미완성' in w for w in q['warnings']))
        with self.assertRaises(ValueError):main.prepare_prices(x,now=x.index[-2])

    def test_metadata_overrides_without_network(self):
        with patch('main.yf.Ticker') as ticker:
            r=main.resolve_metadata('AAPL','Apple','USD')
            ticker.assert_not_called()
            self.assertEqual(r['currency'],'USD')

    def test_metadata_failure_fallback(self):
        with patch('main.yf.Ticker',side_effect=RuntimeError('offline')):
            r=main.resolve_metadata('AAPL',None,None)
            self.assertEqual(r['name'],'AAPL')
            self.assertEqual(r['currency'],'UNKNOWN')

    def test_currency_and_precision(self):
        i=main.indicators(prices());i['macd']=.0123
        c=main.build_context('Apple','AAPL',i,'USD')
        self.assertIn('Prices in USD',c)
        self.assertIn('MACD 0.0123',c)

    def test_retry_then_success(self):
        with patch('main.urllib.request.urlopen',side_effect=[urllib.error.URLError('offline'),io.BytesIO(b'{"ok":true}')]) as call, patch('main.time.sleep'):
            self.assertEqual(main.request_json('http://test',retries=1),{'ok':True})
            self.assertEqual(call.call_count,2)

    def test_invalid_json_no_retry(self):
        with patch('main.urllib.request.urlopen',return_value=io.BytesIO(b'not json')) as call:
            with self.assertRaises(RuntimeError):main.request_json('http://test')
            self.assertEqual(call.call_count,1)

    def test_http_404_no_retry(self):
        error=urllib.error.HTTPError('http://test',404,'missing',None,None)
        with patch('main.urllib.request.urlopen',side_effect=error) as call:
            with self.assertRaises(RuntimeError):main.request_json('http://test')
            self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
