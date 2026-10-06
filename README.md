# nimble-test

야후 파이낸스에서 일봉 시세를 받아 기술적 지표를 계산하고, 로컬 [ollama](https://ollama.com)에 올린 **nimble(Bespoke-Nimble-9B)** 분류 모델로 추세·모멘텀·매매 신호를 판단하는 실험용 스크립트입니다. 기본 대상 종목은 SK하이닉스(`000660.KS`)입니다.

> 신호만 출력하며 **주문은 실행하지 않습니다.** 결과는 모델의 분류 결과일 뿐 투자 권유가 아닙니다.

## 동작 방식

1. **시세 수집** — `yfinance`로 수정주가 기준 일봉을 가져옵니다. (최소 130거래일 필요)
2. **지표 계산** — `pandas`로 아래 지표를 직접 계산합니다.
   - 수익률: 1 / 5 / 20 / 60일
   - 이동평균: SMA 5 / 20 / 60 / 120
   - RSI(14, Wilder 방식)
   - MACD(12, 26, 9)와 히스토그램(당일·전일)
   - 볼린저 밴드(20, 2)와 %B
   - 거래량 및 20일 평균 대비 배율
   - 52주 고가·저가
3. **컨텍스트 생성** — 지표를 영어 문장으로 풀어 쓴 텍스트를 만듭니다.
4. **분류** — 스키마의 필드마다 모델을 한 번씩 호출해 선택지별 확률을 구합니다.

### 분류 스키마

| 필드 | 선택지 | 의미 |
| --- | --- | --- |
| `trend` | `uptrend` / `sideways` / `downtrend` | 이동평균과 최근 수익률이 가리키는 추세 방향 |
| `momentum` | `overbought` / `bullish` / `neutral` / `bearish` / `oversold` | RSI·MACD·볼린저 %B 기준 모멘텀 상태 |
| `action` | `BUY` / `HOLD` / `SELL` | 지표 기반 당일 매매 판단 |

### nimble 모델 호출 방식

nimble은 채팅 모델이 아니라 **"컨텍스트 + 스키마 → 선택지 코드 한 글자"** 를 내는 분류 모델입니다. 그래서 일반적인 chat API 대신 다음과 같이 호출합니다.

- ollama 쪽 `TEMPLATE`이 `{{ .Prompt }}` 뿐이므로 ChatML 프롬프트를 직접 조립해 `/api/generate`에 `raw: true`로 보냅니다.
- 프롬프트 형식은 Bespoke-Nimble의 `parallel_schema.prepare_prompts`와 동일하며, 시스템 프롬프트는 학습 시 사용된 것(`schema_config.json`)과 같아야 합니다.
- 각 선택지에는 `A`, `B`, `C`… 한 글자 코드가 순서대로 부여됩니다.
- `num_predict: 1`, `temperature: 0`으로 토큰 하나만 생성하고, 첫 토큰의 `top_logprobs`(상위 20개)에서 후보 코드의 logprob만 골라 정규화해 선택지별 확률을 계산합니다. 상위 20개 밖의 후보는 확률 0으로 취급합니다.

## 요구 사항

- Python 3.11 이상
- [uv](https://docs.astral.sh/uv/)
- 실행 중인 ollama 서버(기본 `http://localhost:11434`)와 `nimble`이라는 이름으로 등록된 Bespoke-Nimble-9B 모델
  - `/api/generate`의 `logprobs` / `top_logprobs` 옵션을 지원하는 ollama 버전이어야 합니다.
  - 모델의 `TEMPLATE`은 `{{ .Prompt }}`로 설정되어 있어야 합니다.
- 야후 파이낸스에 접속할 수 있는 네트워크

## 설치

```bash
uv sync
```

## 실행

```bash
uv run main.py
```

다른 종목을 보려면 야후 파이낸스 티커와 표시 이름을 지정합니다.

```bash
uv run main.py --ticker 005930.KS --name "Samsung Electronics"
```

결과를 JSON으로 받으려면 `--json`을 붙입니다. 계산된 지표, 모델에 전달한 컨텍스트, 필드별 예측과 확률이 모두 포함됩니다.

```bash
uv run main.py --json
```

### 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--ticker` | `000660.KS` | 야후 파이낸스 티커 |
| `--name` | `SK hynix` | 컨텍스트와 출력에 쓰이는 종목 이름 |
| `--period` | `2y` | 수집 기간 (yfinance `period`) |
| `--model` | `nimble` | ollama 모델 이름 |
| `--host` | `http://localhost:11434` | ollama 서버 주소 |
| `--json` | 꺼짐 | 결과를 JSON으로 출력 |

### 출력 형식

기본 출력은 기준일, 모델에 전달한 컨텍스트, 필드별 예측과 확률 분포, 최종 신호 순서입니다. (아래 `…` 자리에 실제 값이 들어갑니다.)

```text
[SK hynix 000660.KS] YYYY-MM-DD 기준

Stock: SK hynix (000660.KS), daily data as of YYYY-MM-DD. Prices in KRW.
Close … (open …, high …, low …).
…

--- nimble 판단 ---
trend    : …          (uptrend …%  sideways …%  downtrend …%)
momentum : …          (overbought …%  bullish …%  neutral …%  bearish …%  oversold …%)
action   : …          (BUY …%  HOLD …%  SELL …%)

=> 신호: … (확률 …%)
※ 모델의 분류 결과일 뿐이며 투자 권유가 아닙니다. 주문은 실행되지 않습니다.
```

## 프로젝트 구조

```text
.
├── main.py          # 시세 수집, 지표 계산, 프롬프트 생성, 분류, CLI
├── pyproject.toml   # 프로젝트 메타데이터와 의존성 (pandas, yfinance)
└── uv.lock          # 잠긴 의존성 버전
```

`main.py`의 주요 함수:

| 함수 | 역할 |
| --- | --- |
| `fetch_prices` | yfinance로 일봉 수집, 데이터 부족 시 종료 |
| `indicators` | 기술적 지표 계산 |
| `build_context` | 지표를 모델 입력용 영어 텍스트로 변환 |
| `build_prompt` | 컨텍스트·스키마·요청 필드로 ChatML 프롬프트 조립 |
| `classify` | ollama 호출 후 첫 토큰 logprob으로 선택지별 확률 계산 |
| `main` | CLI 인자 처리와 결과 출력 |

## 참고 사항

- 컨텍스트의 가격 단위는 `KRW`로 고정되어 있어, 원화가 아닌 종목에 쓰려면 `build_context`를 수정해야 합니다.
- 일봉이 130개 미만이면 종료되므로 `--period`를 충분히 길게 잡아야 합니다. (SMA120 계산에 필요)
- 스키마를 바꾸려면 `main.py`의 `SCHEMA`를 수정합니다. 필드당 모델 호출이 한 번씩 발생합니다.
- 모델이 선택지 코드를 상위 20개 토큰 안에 내지 않으면 해당 필드에서 오류로 중단됩니다.
