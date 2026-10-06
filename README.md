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

- 수정주가 기준 일봉 최소 130개가 필요합니다. 결측·무한대·잘못된 OHLC 관계·음수 거래량·중복 또는 역순 날짜는 거부합니다.
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
- 후보 코드가 top_logprobs에서 누락
- 후보 확률 질량이 0.5 미만이거나 1을 초과

0.5는 실험용 휴리스틱이며 성능 검증으로 보정된 기준이 아닙니다. 응답 구조가 잘못되면 오류로 종료합니다. 누락된 후보는 `missing_codes`에 기록하고, 유효하지 않은 응답을 BUY/HOLD/SELL로 강제 변환하지 않습니다.

## 실행 기록

`--output-dir runs`를 지정하면 UUID별 디렉터리에 다음 파일을 저장합니다.

- `prices.csv`: 실제 지표 계산에 사용한 수정 OHLCV 스냅샷
- `result.json`: 기준일·데이터 경고·설정·지표·컨텍스트·분류 결과·실행 시각
- `main.py`: 실행 코드 스냅샷

JSON에는 스키마, 시스템 프롬프트, 필드별 프롬프트 해시, 입력 CSV 해시, 코드 해시와 Python/pandas/yfinance 버전도 포함됩니다. `/api/tags`로 모델 digest를 조회하며 실패하면 미확인 상태와 경고를 기록합니다. digest는 추론 이후 조회한 값이므로 실행 중 서버의 모델이 교체되지 않았다는 보장은 없습니다. `runs/`는 Git에서 제외됩니다.

## 과거 시점별 평가

고정 CSV를 사용해 각 기준일까지의 데이터로 지표와 컨텍스트를 만들고, 규칙 기반 신호 또는 모델의 action을 비교합니다.

```bash
uv run evaluate.py --prices runs/RUN_ID/prices.csv --ticker AAPL --name Apple --currency USD
uv run evaluate.py --prices runs/RUN_ID/prices.csv --ticker AAPL --currency USD --with-model --horizon 5 --step 5 --threshold 1
```

- `--with-model`이 없으면 네트워크 없이 규칙 기반 평가만 실행합니다.
- `--horizon`은 미래 관측 거래일 수, `--step`은 평가 간격, `--threshold`는 중립 구간의 퍼센트 기준입니다.
- 다음 거래일 시가부터 horizon번째 거래일 종가까지의 가격 변화가 threshold 초과면 BUY, 음의 threshold 미만이면 SELL, 나머지는 HOLD 라벨입니다.
- 규칙은 종가/SMA20/SMA60의 정렬, MACD 히스토그램 부호와 RSI 구간으로 BUY/SELL을 판단하며 나머지는 HOLD입니다.
- 결과는 방향 라벨 일치율과 모델 판단 비율(coverage)입니다. 보류한 모델 결과는 모델 일치율 분모에서 제외하며 coverage를 함께 보고합니다.
- 기본 저장 위치는 `runs/evaluations/UUID/`입니다. 평가 결과, 설정, 입력 스냅샷과 두 Python 코드도 저장합니다.
- 저장된 `result.json`에서 시장 시간대를 읽습니다. 독립 CSV는 `--timezone Asia/Seoul`처럼 지정할 수 있습니다.

이는 포트폴리오 수익률 백테스트가 아닙니다. 거래비용·슬리피지·보유 포지션을 반영하지 않으며, step < horizon이면 관측 구간이 겹칩니다. 수정주가의 사후 수정과 모델 학습 데이터의 미래 정보 또는 과거 데이터 기억도 통제하지 않습니다.

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
