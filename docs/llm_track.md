# LLM 연구 트랙

이 트랙의 목적은 큰 점수를 얻는 것이 아니라, 같은 작은 문제를 서로 다른 방법으로
학습하며 **무엇이 바뀌고 왜 바뀌는지 눈으로 확인하는 것**이다. 모든 tiny 실험은
CPU와 오프라인 환경에서 실행되며 모델 다운로드가 없다. 실제 모델 워크플로는
`${FMLAB_MODEL_ROOT}/Qwen` 아래 체크포인트만 `local_files_only=True`로 읽는다.

## 구현 범위

| 주제 | 실제 구현 | 눈으로 보는 결과 |
|---|---|---|
| tokenizer | UTF-8 byte tokenizer, causal/SFT dataset, prompt loss mask | token label, 입력/label mask |
| from scratch | decoder-only Transformer | loss PNG, 생성 sample, attention SVG |
| attention | causal MHA와 GQA, head별 weight 반환 | query×key heatmap |
| 현대 LLM block | RoPE, RMSNorm, SwiGLU, tied embedding | 모듈별 코드와 parameter count |
| continued pretraining | 일반 corpus 모델을 domain 원문으로 추가 학습 | domain loss 전후 |
| full SFT | assistant response token에만 cross-entropy | SFT loss curve |
| LoRA | base freeze, A/B 저랭크 행렬 주입, adapter state | trainable parameter 비율 |
| QLoRA | frozen INT4/INT8 dequantized base + LoRA | LoRA와 quantization error 동시 관찰 |
| quantization | per-channel symmetric INT8/INT4 simulation | 압축률, weight MAE/RMSE/cosine, logit/loss 차이 |
| distillation | temperature KL + hard-label CE | soft/hard/total loss trace |
| DPO | reference-relative chosen/rejected logistic loss | reward accuracy/margin, loss trace |
| RAG | BM25, callback 기반 dense cosine retrieval | Recall@k, Hit@k, MRR, query별 순위 |
| 로컬 Qwen | CPT/full SFT/LoRA/QLoRA/KD/quantization/DPO adapters | JSON metrics와 선택적 checkpoint |

핵심 파일은 다음과 같다.

- `data.py`: 다운로드 없는 tokenizer와 학습 데이터 구성
- `model.py`: decoder block 전체 구현과 attention hook
- `training.py`: 재현 가능한 작은 optimizer loop
- `lora.py`, `quantization.py`: LoRA/QLoRA 및 양자화 원리 구현
- `distillation.py`, `preference.py`: KD와 DPO loss
- `retrieval.py`: BM25/dense RAG 검색 및 평가
- `experiments.py`, `smoke.py`: 결과물까지 생성하는 end-to-end 실행
- `workflows.py`, `dpo_workflow.py`: 로컬 Qwen용 선택 의존성 workflow

## 가장 먼저 실행할 실험

기본 의존성과 개발 의존성을 준비한 뒤 아래처럼 실행한다.

```bash
cd /path/to/foundation-model-lab
python -m pip install -e '.[dev]'

python - <<'PY'
from fmlab.llm.experiments import run_tiny_from_scratch

result = run_tiny_from_scratch(
    "${FMLAB_DATA_ROOT}/artifacts/llm/tiny-from-scratch"
)
print(result.to_dict())
PY
```

브라우저에서 결과 디렉터리의 `report.html`을 열면 된다. `loss_curve.png`는 학습이
진행되는 모습, `attention_layer0_head0.svg`는 한 head가 이전 token 중 어디를 보는지,
`generated_sample.txt`는 짧은 생성 결과를 보여준다. SVG의 각 cell에 마우스를 올리면
정확한 attention weight가 표시된다.

전체 objective를 1분 이내 toy run으로 훑으려면 다음을 실행한다.

```bash
python - <<'PY'
from fmlab.llm.smoke import run_offline_llm_smoke_suite

result = run_offline_llm_smoke_suite(
    "${FMLAB_DATA_ROOT}/artifacts/llm/offline-suite",
    steps=3,
    device="cpu",
)
print(result.to_dict())
PY
```

이 실행은 순서대로 from-scratch pretraining, domain continued pretraining, full SFT,
LoRA, simulated QLoRA, INT8/INT4, teacher/student KD, DPO, BM25 평가를 실제로 수행한다.
Tiny 모델은 원리를 확인하기 위한 것이므로 생성 문장 품질은 평가 대상이 아니다.

## 검증 실행에서 관찰된 결과

2026-08-01에 CPU, objective당 2 step으로 실행한 smoke 결과다. seed는 7이다.

| 관찰값 | 결과 | 해석 |
|---|---:|---|
| from-scratch loss | 5.5405 → 5.2449 | next-token optimizer가 정상적으로 학습함 |
| domain continued-pretraining loss | 5.5947 → 5.5085 | 새 법률 문체 corpus에 적응하기 시작함 |
| LoRA trainable parameters | 256 / 18,912 (1.35%) | base 대부분을 고정하고도 gradient가 흐름 |
| INT8 추정 압축률 | 3.58× | scale metadata 때문에 정확히 4×는 아님 |
| INT8 cosine similarity | 0.999986 | 작은 weight 왜곡 |
| INT4 추정 압축률 | 6.47× | scale metadata 때문에 정확히 8×는 아님 |
| INT4 cosine similarity | 0.995322 | 압축과 함께 INT8보다 큰 왜곡 |
| INT4 weight RMSE | 0.002020 | INT8의 0.000112보다 큼 |
| distillation total loss | 1.6751 → 1.6550 | student가 teacher 분포 쪽으로 이동함 |
| DPO loss | 0.6931 → 0.6611 | 동일 policy/reference에서 출발해 chosen을 선호하기 시작함 |
| BM25 Recall@2 / MRR | 0.50 / 0.50 | 단순 한국어 어절 tokenization의 한계를 드러냄 |
| 전체 실행 시간 | 약 0.77초 | CPU에서도 반복 실험 가능한 크기 |

BM25 점수 0.5는 실패를 숨기지 않은 의도적인 학습 포인트다. 현재 tokenizer는 공백
기반 한국어 어절을 그대로 사용하므로 `비트`와 `비트로`를 다른 token으로 본다.
형태소 analyzer, character n-gram, dense embedding을 바꾸고 Recall/MRR이 어떻게
변하는지 비교하는 것이 다음 ablation이다.

## 각 실험에서 볼 것

### 1. Tokenizer와 from-scratch Transformer

`ByteTokenizer`는 UTF-8 byte 하나를 token 하나로 둔다. vocabulary가 260개라 완전히
투명하고 어떤 한국어도 `<unk>`가 되지 않지만, 한글 한 글자가 보통 3 token이어서
sequence가 길어진다. 이후 SentencePiece/BPE와 다음을 비교하면 된다.

- 평균 tokens/문장, 한국어 글자당 token 수
- vocabulary 크기에 따른 embedding parameter 수
- 같은 model/context에서 validation loss
- code, 숫자, 조사와 어미의 분할 방식

`TinyDecoderLM`은 입력 embedding → 반복 decoder block → RMSNorm → LM head 구조다.
각 block은 pre-norm causal attention과 SwiGLU FFN을 residual로 더한다. 위치 정보는
Q/K vector에 RoPE로 적용한다. `return_attentions=True`가 각 layer의
`[batch, head, query, key]` 행렬을 반환한다.

추천 ablation은 `n_heads`, `n_kv_heads`, layer 수, 폭, context length, RoPE base를 한
번에 하나만 변경하는 것이다. parameter 수를 맞춘 깊은 모델과 넓은 모델 비교도 좋다.

### 2. Continued pretraining과 full SFT

Continued pretraining은 원문 전체를 next-token label로 사용한다. 용어·문체·도메인
분포를 배우지만 질문에 답하는 형식을 직접 가르치지는 않는다. Full SFT는
`InstructionDataset`에서 prompt label을 `-100`으로 가려 assistant response만
학습한다. 다음 네 지표를 함께 본다.

- domain validation perplexity
- general validation perplexity(망각 확인)
- instruction 정답률/형식 준수율
- peak VRAM과 step time

### 3. LoRA와 QLoRA

`inject_lora`는 선택한 linear layer를 `base(x) + scale·B·A·x`로 바꾸고 base를
고정한다. B를 zero로 초기화하므로 주입 직후 출력이 원본과 같다. rank와 target을
바꾸어 trainable parameter 수 대비 성능을 그린다.

Toy QLoRA는 quantized value를 int8 container에 보관해 forward 때 dequantize한다.
INT4 packed CUDA kernel의 실제 메모리/속도 구현이 아니라 **수치 오차와 gradient
경로를 관찰하는 simulation**이다. 실제 Qwen QLoRA는 bitsandbytes NF4 설정을 쓴다.

### 4. Quantization

`benchmark_weight_quantization`은 matrix weight를 output channel별로 quantize하고
packed byte 추정치, MAE, RMSE, cosine similarity를 기록한다. INT4가 작아지는 대신
오차가 커지는지 먼저 확인한 후, 같은 prompt set으로 다음을 기록한다.

- model load 시간과 peak VRAM
- first-token latency와 tokens/s
- task accuracy/perplexity
- 생성 결과 diff

작은 모델은 dequantization overhead 때문에 더 작아져도 반드시 빨라지지는 않는다.

### 5. Knowledge distillation

Student loss는 hard CE와 temperature-scaled KL을 합친다. temperature를 높이면 teacher
분포의 작은 확률 차이가 더 드러난다. `soft_target_weight=0`, `1`, 혼합값을 비교해
정답 label만 배울 때와 teacher의 상대적 선호까지 배울 때를 비교한다.

### 6. DPO

DPO는 chosen/rejected log-probability margin을 reference model의 margin과 비교한다.
처음 policy와 reference가 같으면 loss가 `-log(sigmoid(0)) ≈ 0.6931`이다. 한 step 후
chosen 쪽 margin이 커지면 loss가 내려간다. 데이터에 길이 bias가 생기지 않도록
chosen/rejected 길이 분포와 평균 token log-probability도 함께 확인해야 한다.

### 7. RAG

생성 전에 retrieval만 분리 평가한다. `BM25Retriever`는 외부 패키지 없이 실행되고,
`DenseRetriever`는 로컬 embedding callback만 받는다. relevance label이 있는 query로
Recall@k, Hit@k, MRR을 측정한 다음에만 `build_grounded_prompt`로 생성 모델을 붙인다.
이 순서가 검색 실패와 LLM hallucination을 구분해 준다.

## 로컬 Qwen 워크플로 실행

선택 의존성을 설치한다. Blackwell/CUDA 환경에서 현재 PyTorch를 유지한 채 설치해야
하며, 기존 모델은 복사하지 않는다.

```bash
python -m pip install -e '.[llm]'
```

설정은 `configs/llm/`에 있다. 모든 설정은 기본 `smoke: true`, `max_steps: 2`,
`save_model: false`다. 따라서 첫 실행에서 대형 checkpoint를 새로 저장하지 않는다.

Continued pretraining/full SFT/LoRA/QLoRA:

```python
from fmlab.config import load_config
from fmlab.llm.workflows import LocalWorkflowConfig, run_local_training, workflow_plan

raw = load_config("configs/llm/lora.yaml")
config = LocalWorkflowConfig.from_mapping(raw)
print(workflow_plan(config))
metrics = run_local_training(config)
```

Teacher/student distillation:

```python
from fmlab.config import load_config
from fmlab.llm.workflows import LocalWorkflowConfig, run_local_distillation

config = LocalWorkflowConfig.from_mapping(load_config("configs/llm/distillation.yaml"))
metrics = run_local_distillation(config)
```

INT4/INT8 한 precision의 실제 generation benchmark:

```python
from fmlab.config import load_config
from fmlab.llm.workflows import LocalWorkflowConfig, run_local_generation_benchmark

config = LocalWorkflowConfig.from_mapping(load_config("configs/llm/quantization.yaml"))
metrics = run_local_generation_benchmark(config)
```

DPO:

```python
from fmlab.config import load_config
from fmlab.llm.dpo_workflow import run_local_dpo
from fmlab.llm.workflows import LocalWorkflowConfig

config = LocalWorkflowConfig.from_mapping(load_config("configs/llm/dpo.yaml"))
metrics = run_local_dpo(config)
```

설치되지 않은 선택 패키지가 있으면 `OptionalDependencyError`가 필요한 extra 설치
명령과 함께 발생한다. CUDA가 없는 환경에서 bitsandbytes QLoRA를 요청한 경우도
가짜 CPU fallback을 하지 않고 이유를 명확히 알린다.

## 데이터 형식

Continued pretraining JSONL:

```json
{"text": "도메인 원문 한 문단"}
```

SFT/LoRA/QLoRA JSONL:

```json
{"prompt": "질문", "response": "학습할 답변"}
```

또는 tokenizer의 chat template를 이용할 수 있다.

```json
{"messages": [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}
```

DPO JSONL:

```json
{"prompt": "질문", "chosen": "선호 답변", "rejected": "비선호 답변"}
```

## 안전한 확장 순서

1. `save_model: false`, 2 step으로 model/tokenizer/VRAM 호환성을 확인한다.
2. 고정 validation set으로 20–100 step을 실행해 loss 방향을 본다.
3. 데이터 leakage와 label mask를 눈으로 검사한다.
4. 한 변수만 바꾼 ablation을 최소 3 seed로 실행한다.
5. best adapter만 저장하고 full merged model은 필요할 때 한 번만 만든다.
6. 품질, peak VRAM, step time, artifact byte를 항상 같은 report에 기록한다.

테스트는 다음 명령으로 실행한다.

```bash
PYTHONPATH=src pytest -q tests/test_llm_core.py tests/test_llm_objectives.py
```

현재 구현 검증에서는 12개 테스트가 모두 통과했다.

