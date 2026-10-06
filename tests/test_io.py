import io
import unittest
from unittest.mock import patch
import urllib.error
import pandas as pd
import main
from test_core import prices


class IOTests(unittest.TestCase):
    def test_valid_prices_skip_repair(self):
        with patch('main.yf.Ticker') as ticker:
            ticker.return_value.history.return_value=prices()
            result=main.fetch_prices('TEST','2y')
            self.assertEqual(ticker.return_value.history.call_count,1)
            self.assertFalse(result.attrs['recovery']['attempted'])

    def test_ohlc_repair_success_and_audit_survives(self):
        bad=prices(131);bad.iloc[0,2]=101
        fixed=prices(131)
        with patch('main.yf.Ticker') as ticker, patch('builtins.print'):
            ticker.return_value.history.side_effect=[bad,fixed]
            df=main.fetch_prices('TEST','2y')
            self.assertTrue(ticker.return_value.history.call_args.kwargs['repair'])
            self.assertEqual(df.attrs['recovery']['changed_rows'],1)
            self.assertEqual(df.attrs['recovery']['changes'][0]['before']['Low'],101)
            _,q=main.prepare_prices(df,now=df.index[-1])
            self.assertEqual(q['recovery']['status'],'recovered')
            self.assertTrue(any('복구' in w for w in q['warnings']))

    def test_ohlc_repair_unresolved(self):
        bad=prices();bad.iloc[0,2]=101
        with patch('main.yf.Ticker') as ticker, patch('builtins.print'):
            ticker.return_value.history.side_effect=[bad,bad]
            with self.assertRaisesRegex(RuntimeError,'복구 후에도 검증 실패'):
                main.fetch_prices('TEST','2y')
            self.assertEqual(ticker.return_value.history.call_count,2)

    def test_repair_network_failure(self):
        bad=prices();bad.iloc[0,2]=101
        with patch('main.yf.Ticker') as ticker, patch('builtins.print'):
            ticker.return_value.history.side_effect=[bad,RuntimeError('offline')]
            with self.assertRaisesRegex(RuntimeError,'복구 조회 실패'):
                main.fetch_prices('TEST','2y')

    def test_missing_price_repair(self):
        bad=prices();bad.iloc[0,0]=float('nan')
        with patch('main.yf.Ticker') as ticker, patch('builtins.print'):
            ticker.return_value.history.side_effect=[bad,prices()]
            df=main.fetch_prices('TEST','2y')
            self.assertIsNone(df.attrs['recovery']['changes'][0]['before']['Open'])

    def test_short_history_not_retried(self):
        with patch('main.yf.Ticker') as ticker:
            ticker.return_value.history.return_value=prices(20)
            with self.assertRaises(ValueError):main.fetch_prices('TEST','1mo')
            self.assertEqual(ticker.return_value.history.call_count,1)

    def test_repair_added_dates_excluded_and_logged(self):
        bad=prices();bad.iloc[0,2]=101
        repaired=prices(131)
        with patch('main.yf.Ticker') as ticker, patch('builtins.print'):
            ticker.return_value.history.side_effect=[bad,repaired]
            result=main.fetch_prices('TEST','2y')
            self.assertTrue(result.index.equals(bad.index))
            self.assertEqual(result.attrs['recovery']['excluded_added_dates'],[repaired.index[-1].isoformat()])

    def test_repair_cannot_drop_invalid_row(self):
        bad=prices(131);bad.iloc[0,2]=101
        with patch('main.yf.Ticker') as ticker, patch('builtins.print'):
            ticker.return_value.history.side_effect=[bad,bad.iloc[1:]]
            with self.assertRaisesRegex(RuntimeError,'날짜가 달라졌습니다'):
                main.fetch_prices('TEST','2y')

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
