# nimble-test

Yahoo Finance 일봉으로 기술적 지표를 계산하고, 로컬 Ollama의 Nimble 분류 모델로 추세·모멘텀·매매 신호를 분류하는 실험용 Python 스크립트입니다. 기본 티커는 SK하이닉스(`000660.KS`)입니다. 주문은 실행하지 않습니다.

## 설치와 실행

Python 3.11 이상, uv, `nimble` 모델을 등록한 Ollama 서버가 필요합니다. 서버는 `/api/generate`의 `logprobs`와 `top_logprobs`를 지원해야 합니다. 모델 TEMPLATE은 `{{ .Prompt }}`를 사용하며 ChatML 프롬프트를 `raw: true`로 전달합니다. 모델별 템플릿 호환성은 실제 서버에서 확인해야 합니다.

```bash
uv sync
uv run main.py
uv run main.py --ticker AAPL --name Apple --currency USD --json --output-dir runs
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ticker` | `000660.KS` | Yahoo Finance 티커 |
| `--name` | 자동 | 메타데이터 이름, 조회 실패 시 티커 사용 |
| `--currency` | 자동 | 메타데이터 통화, 조회 실패 시 UNKNOWN 및 경고 |
| `--period` | `2y` | 수집 기간 |
| `--model` | `nimble` | Ollama 모델 이름 |
| `--host` | `http://localhost:11434` | Ollama 서버 주소 |
| `--timeout` | `60` | 각 Ollama 요청 제한 시간(초) |
| `--retries` | `1` | 연결 오류 및 HTTP 429/일부 5xx 재시도, 최대 3회 |
| `--include-current-day` | 꺼짐 | 미완성 가능성이 있는 당일 일봉 포함 |
| `--json` | 꺼짐 | 엄격한 JSON 결과 출력; 경고는 stderr |
| `--output-dir` | 저장 안 함 | 실행 기록과 입력 시세 저장 디렉터리 |

## 데이터와 지표 정책

- 수정주가 기준 일봉 최소 130개가 필요합니다. 결측·무한대·잘못된 OHLC 관계·음수 거래량 등 값 오류가 발견되면 `yfinance.history(repair=True)`로 한 번 복구 조회하고 재검증합니다. 복구 의존성 scipy와 scikit-learn은 프로젝트에 포함됩니다.
- 복구 후에도 값이 잘못됐거나 원래 일봉 날짜가 누락됐으면 명확한 오류로 종료합니다. 잘못된 행을 임의로 삭제하거나 고가·저가를 강제 보정하지 않습니다. 데이터 부족·중복·역순 날짜는 복구 대상이 아닙니다.
- 복구 시도/성공은 stderr에 표시하며, 복구가 추가한 날짜는 사용하지 않고 `excluded_added_dates`에 기록합니다. 성공한 복구의 원래 오류·변경 일봉 수·변경 전후 OHLCV를 JSON의 `data_quality.recovery`에 기록합니다. OHLC 오류 메시지에는 문제 일봉 수와 최대 3개 날짜/가격 예시를 표시합니다.
- 복구는 성공을 보장하지 않습니다. Yahoo의 모든 유형의 잘못된 가격을 yfinance가 수정할 수 있는 것은 아닙니다.
- 시장 시간대의 당일 일봉은 기본 제외합니다. 장 마감 이후에도 당일은 제외하므로 필요하면 `--include-current-day`를 사용하세요. 시간대가 없는 데이터는 UTC 기준으로 처리하고 경고합니다.
- 기준일이 7일 넘게 오래되면 경고합니다. 거래소 휴장일 달력은 적용하지 않습니다.
- 수익률 1/5/20/60일, SMA 5/20/60/120, MACD 12/26/9, 거래량 배율 등을 계산합니다.
- RSI(14)는 첫 14개 변화량의 산술평균으로 초기화한 Wilder 평활을 사용합니다. 완전 일정 가격은 RSI=50, 손실 없이 상승한 경우 RSI=100입니다.
- 볼린저 밴드는 20일 평균 ± 표준편차(ddof=0)의 2배입니다. 폭이 0이면 %B=0.5로 정의합니다.
- 직전 20일 평균 거래량이 0이면 배율은 JSON `null`, 컨텍스트에서는 unavailable로 표시합니다.
- 52주 범위는 기준일 이전 1년의 수정 High/Low입니다. 전체 1년의 입력 이력이 없으면 실제 관측 일봉 수와 incomplete year를 표시합니다.
- 가격·이동평균·MACD는 컨텍스트에 소수점 4자리로 전달합니다.

## 분류 결과 해석

`trend`, `momentum`, `action` 필드마다 한 번씩 모델을 호출합니다. 각 필드는 A~Z 코드로 선택지를 표현하며 한 토큰을 생성합니다.

결과의 `probabilities`는 관측된 후보 토큰 사이의 **조건부 상대 비중**입니다. 실제 분류 정확도나 수익 확률을 뜻하지 않습니다. `candidate_mass`는 후보 토큰의 원래 확률 합입니다.

다음 경우 `prediction: null`, `status: abstained`로 판단을 보류합니다.

- 생성 코드가 허용 선택지에 없거나 가장 높은 후보와 불일치
- 후보 코드가 top_logprobs에서 누락하거나 같은 후보 코드가 중복되어 모호함
- 후보 확률 질량이 0.5 미만이거나 1을 초과

0.5는 실험용 휴리스틱이며 성능 검증으로 보정된 기준이 아닙니다. 선택지 밖의 토큰은 빈 문자열 등 같은 문자열로 여러 번 표시되어도 허용합니다. `top_token_mass`는 원본 항목별 확률을 모두 합산하므로 중복 문자열의 질량도 빠뜨리지 않습니다. 후보 코드가 중복되면 `ambiguous_candidate_tokens`로 판단을 보류하고 `duplicate_codes`에 기록합니다. 응답 구조가 잘못되면 오류로 종료합니다. 누락된 후보는 `missing_codes`에 기록하고, 유효하지 않은 응답을 BUY/HOLD/SELL로 강제 변환하지 않습니다.

## 실행 기록

`--output-dir runs`를 지정하면 UUID별 디렉터리에 다음 파일을 저장합니다.

- `prices.csv`: 실제 지표 계산에 사용한 수정 OHLCV 스냅샷
- `result.json`: 기준일·데이터 경고·설정·지표·컨텍스트·분류 결과·실행 시각
- `main.py`: 실행 코드 스냅샷

JSON에는 스키마, 시스템 프롬프트, 필드별 프롬프트 해시, 입력 CSV 해시, 코드 해시와 Python/pandas/yfinance 버전도 포함됩니다. `/api/tags`로 모델 digest를 조회하며 실패하면 미확인 상태와 경고를 기록합니다. digest는 추론 이후 조회한 값이므로 실행 중 서버의 모델이 교체되지 않았다는 보장은 없습니다. `runs/`는 Git에서 제외됩니다.

## 과거 시점별 평가

고정 CSV를 사용해 각 기준일까지의 데이터로 지표와 컨텍스트를 만들고, 규칙 기반 신호 또는 모델의 방향 예측을 비교합니다. 매매 행동은 --objective action으로 별도 평가합니다.

```bash
uv run evaluate.py --prices runs/RUN_ID/prices.csv --ticker AAPL --name Apple --currency USD
uv run evaluate.py --prices runs/RUN_ID/prices.csv --ticker AAPL --currency USD --with-model --horizon 5 --step 5 --threshold 1
```

- `--with-model`이 없으면 네트워크 없이 규칙 기반 평가만 실행합니다.
- `--horizon`은 미래 관측 거래일 수, `--step`은 평가 간격, `--threshold`는 중립 구간의 퍼센트 기준입니다.
- 다음 거래일 시가부터 horizon번째 거래일 종가까지의 가격 변화가 threshold 초과면 UP, 음의 threshold 미만이면 DOWN, 나머지는 FLAT 라벨입니다. action 모드의 기존 라벨은 BUY/SELL/HOLD의 방향 proxy로 남습니다.
- 규칙은 종가/SMA20/SMA60의 정렬, MACD 히스토그램 부호와 RSI 구간으로 BUY/SELL을 판단하며 나머지는 HOLD입니다.
- 결과는 방향 라벨 일치율과 모델 판단 비율(coverage)입니다. 보류한 모델 결과는 모델 일치율 분모에서 제외하며 coverage를 함께 보고합니다.
- 기본 저장 위치는 `runs/evaluations/UUID/`입니다. 평가 결과, 설정, 입력 스냅샷과 두 Python 코드도 저장합니다.
- 저장된 `result.json`에서 시장 시간대를 읽습니다. 독립 CSV는 `--timezone Asia/Seoul`처럼 지정할 수 있습니다.

direction 결과는 포트폴리오 수익률 백테스트가 아닙니다. action의 비용·포지션 백테스트는 아래 후속 평가 절을 참고하세요. 방향 지표에는 거래비용·슬리피지·보유 포지션을 반영하지 않으며, step < horizon이면 관측 구간이 겹칩니다. 수정주가의 사후 수정과 모델 학습 데이터의 미래 정보 또는 과거 데이터 기억도 통제하지 않습니다.

## 검증

```bash
uv run python -B -m unittest discover -s tests -v
```

테스트는 합성 데이터와 모의 API 응답을 사용합니다. 실제 시세 수집, 실제 모델 호환성 및 투자 성능은 별도 검증 대상입니다.

## 파일 구성

- `main.py`: 수집·데이터 검증·지표·분류·실행 기록·CLI
- `evaluate.py`: 과거 시점별 규칙/모델 평가 CLI
- `tests/`: 회귀 및 통합 테스트
- `PROJECT_REVIEW.md`: 최초 분석과 단계별 완료 기록
- `pyproject.toml`, `uv.lock`: 프로젝트 의존성

## 후속 평가: 방향 분류와 매매 행동 분리

`main.py`와 `evaluate.py`는 기본적으로 요청 필드만 포함한 스키마를 전달합니다. 전체 스키마는 `--schema-scope full`로 비교할 수 있습니다. 실패 사례 4건에서 전체 스키마의 D 생성이 단일 action 스키마에서 사라졌지만, 모든 데이터에서 같은 현상이 보장되는 것은 아닙니다.

평가 CLI의 기본값은 이제 `--objective direction`입니다. 라벨은 UP/FLAT/DOWN이며 미래 horizon과 중립 threshold가 프롬프트에도 명시됩니다. `--objective action`은 BUY/HOLD/SELL 행동을 출력하고 long/cash 백테스트도 저장합니다. action의 기존 방향 일치율은 행동의 정답을 나타내는 것이 아니므로 legacy proxy로 표시됩니다. 방향 신호를 자동으로 매매 신호로 전환하지 않습니다.

```bash
uv run python evaluate.py --prices runs/RUN_ID/prices.csv --ticker 000660.KS --currency KRW --with-model --objective direction --split-date 2026-01-01
uv run python evaluate.py --prices runs/RUN_ID/prices.csv --ticker 000660.KS --currency KRW --with-model --objective action --split-date 2026-01-01 --fee-bps 10 --slippage-bps 5
```

`metrics`에는 고정 라벨 비교, 혼동행렬(행=예측/열=실제), 클래스별 precision/recall/F1, balanced accuracy와 coverage가 있습니다. 모델 지표는 유효 예측 표본에 대한 값이며 같은 표본의 규칙 및 고정 라벨 지표도 함께 제공합니다.

`--split-date` 이전은 development, 이후는 holdout입니다. 미래 관측 종가가 분할 경계를 넘는 개발 표본은 purged로 제외합니다. 지표 계산에는 기준일까지의 전체 과거 시세를 사용할 수 있습니다. 이미 확인한 과거 기간을 분할한다고 새 미관측 검증 자료가 되는 것은 아닙니다.

### 백테스트 가정

- 매수 신호: 전액 주식 진입. 매도 신호: 전액 현금 청산. 공매도 없음.
- HOLD 및 판단 보류: 기존 포지션 유지. 최초 포지션은 현금.
- 체결: 판단 다음 거래일 시가. 구간 마지막에는 종가로 강제 청산.
- 비용: 편도 수수료 10bps, 슬리피지 5bps 기본값. 1bps=0.01%이며 실제 거래비용으로 추정한 값이 아닌 가정입니다.
- 평가 분할마다 현금으로 새로 시작. 종가 기준 equity curve·최대 낙폭·체결 내역·동일 기간 buy-and-hold 비교 저장.
- 세금·현금 이자·배당 현금흐름을 별도 반영하지 않고 수정 OHLC를 사용합니다. 장중 낙폭/실제 유동성도 반영하지 않습니다.

### 여러 종목의 고정 설정 평가

```bash
uv run python benchmark.py --manifest benchmark_manifest.json --with-model
```

매니페스트는 SK하이닉스 기존 CSV, 삼성전자와 Microsoft의 새 2년 시세를 사용합니다. 설정은 실행 전에 고정하며 최적화하지 않습니다. 기본 horizon=5·step=20·threshold=1%, 검증 시작일=2026-01-01입니다. 실행 중 표본 진행 상황을 출력하고 각 종목/목표가 완료될 때마다 summary.json을 저장합니다. 결과는 `runs/benchmarks/UUID/`에 저장됩니다. 기존 CSV 경로가 없다면 매니페스트의 해당 항목을 현재 CSV로 수정하거나 prices 대신 period를 지정하세요.

과거 D 생성 진단을 반복하려면 다음을 사용합니다.

```bash
uv run python diagnose_schema.py --evaluation runs/evaluations/RUN_ID/result.json --samples 6
```

선정된 실패/정상 표본에서 전체·단일·선택지 순서 변경 스키마를 비교하고 원본 응답을 보존합니다. 이는 원인 진단이며 성능 검증이 아닙니다.

## 매매 정책 개선 검증

```bash
uv run python policy_validation.py --manifest policy_validation_manifest.json --with-model
uv run python policy_attribution.py --run-dir runs/policy_validation/RUN_ID
```

첫 명령은 horizon=step=5를 강제하고, 동일 기간·현금 시작에서 다음을 비교합니다.

- 보유 상태 없이 초기 100% 진입
- 보유 상태 포함, 초기 100% 진입
- 보유 상태 포함, 초기 50% 진입

현재 보유 여부·비중·진입 수정가격·미실현 변화는 해당 판단일 종가까지의 체결과 가격으로 계산합니다. 신호는 다음 거래일 시가에 실행합니다. HOLD/보류/반복 BUY에는 포지션을 유지하며, cap은 초기 진입 비중이므로 가격 변화에 따른 비중 변화는 허용합니다. 매 5일 재검토하는 정책이며 강제 5일 청산 정책은 아닙니다.

두 번째 명령은 같은 신호를 50% 비중으로 다시 실행해 포지션 정책 효과를 분리합니다. 리플레이에는 모델을 다시 호출하지 않습니다. 비용 제거 리플레이도 원래 신호를 고정하므로 비용 변화에 따른 보유 상태·모델 응답 피드백을 재계산한 결과는 아닙니다. 50% 주식/50% 현금의 수동 보유 비교도 포함됩니다.

기본 매니페스트는 2026-09-17까지의 마지막 40거래일을 고정해 기존 3종목과 신규 SPY를 비교합니다. 기존 CSV 경로는 로컬 실행 기록에 의존하므로 다른 환경에서는 해당 파일을 준비하거나 항목을 period 방식으로 바꾸세요. 현재 실행은 과거 자료에 대한 탐색적 검증입니다.

`prospective_policy_plan.json`은 2026-10-06 이후 새로 발생하는 완결 일봉을 대상으로 한 별도 검증 계획입니다. 첫 예측 전에 코드/스키마를 고정하고, 결과를 알기 전에 실제 판단 시각·입력·응답·계좌 상태를 기록해야 합니다. 현재 자동 예약이나 주문 실행은 설정하지 않았습니다.
