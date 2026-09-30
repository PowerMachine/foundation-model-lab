# VLM response distillation: resource-bounded 32B → 8B

이 프로젝트는 Qwen3-VL-32B-Instruct-FP8의 응답을 디스크에 먼저 고정하고, teacher
프로세스가 종료된 뒤 Qwen3-VL-8B LoRA student를 별도 프로세스에서 학습하는
**sequence-level response distillation**이다. teacher logits는 저장하지 않고 두 모델을
GPU에 동시에 올리지 않는다.

- 구현: [`response_distillation.py`](../src/fmlab/vlm/response_distillation.py)
- 실행 계약: [`response_distillation_e2e.yaml`](../configs/vlm/response_distillation_e2e.yaml)
- held-out evaluator: [`distillation_evaluation.py`](../src/fmlab/vlm/distillation_evaluation.py)
- 최신 증거: [2026-08-03 VLM 결과](results_2026-08-03-vlm.md)

```text
canonical manifest
      │
      ├── image-SHA train/eval assignment
      │
      ▼
process 1: 32B FP8 teacher
      │
      ├── resource preflight
      ├── one-request preflight
      └── atomic raw cache + provenance + contract SHA
      │
      ▼
process exit / VRAM release
      │
      ▼
process 2: cache audit
      │
      ├── raw-cache SHA verification
      ├── simulated/type/content/image/split filters
      └── accepted + rejected + audit JSON
      │
      ▼
process 3: 8B student LoRA
      │
      ├── accepted-cache SHA + row-count verification
      ├── train rows only
      └── adapter + metrics + provenance
      │
      ▼
separate generation jobs → CPU paired held-out evaluator
```

## 1. 연구 질문과 주장 경계

핵심 질문은 “teacher response가 같은 gold-label 데이터만 사용한 student보다 도움이
되는가?”다. 최소 세 조건을 같은 frozen holdout에서 비교한다.

| 조건 | 역할 |
|---|---|
| 8B zero-shot | 학습 없는 기준선 |
| 8B gold-label LoRA | 데이터 자체와 LoRA의 효과 |
| 8B teacher-response LoRA | teacher supervision의 추가 효과 |
| 32B teacher, 선택 | student가 teacher gap을 얼마나 회복하는지 분석 |

cache acceptance rate, teacher/reference agreement, student training loss는 held-out student
품질이 아니다. 실제 prediction을 한 seed로 비교해도 결과 수준은 `heldout_comparison`이며,
등록된 최소 3-seed 프로토콜 없이는 **Experiment**라고 부르지 않는다.

## 2. 안전한 full wiring run

model weight를 적재하지 않고 60개 전체에서 schema, split, resume/cache, filtering, student
loop, provenance, SVG/HTML을 검증한다.

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python -m fmlab.vlm.response_distillation mock \
  --manifest "$FMLAB_DATA_ROOT/datasets/vlm/mixed/manifest.jsonl" \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/response-distillation-mock-full" \
  --max-examples 60 \
  --max-steps 5
```

확인된 wiring 결과:

| 항목 | 값 |
|---|---:|
| raw / accepted simulated rows | 60 / 60 |
| train / eval examples | 45 / 15 |
| train / eval image-SHA groups | 9 / 3 |
| image overlap | 0 |
| document / chart / grounding | 24 / 20 / 16 |
| mock loss | `2.052309 → 1.115530` |
| `quality_evaluated` | `false` |

mock 명령만 내부적으로 simulated target을 허용한다. 이 loss는 Qwen3-VL 학습 결과가
아니며 model card나 포트폴리오 성능표에 쓰면 안 된다.

## 3. Teacher-cache 계약

### 입력과 분할

loader는 문서·차트의 nested QA와 grounding row를 atomic example로 정규화한다. 각 example의
identity는 task, image SHA-256, 정규화된 question으로 만든다. train/eval은 row id나 file
path가 아니라 image SHA-256을 seed 17로 묶는다. 같은 pixel에 달린 여러 질문은 항상 같은
split에 남는다.

### Resume contract

재개 가능한 cache가 “이어 쓰기”라는 이유로 서로 다른 실험을 섞지 않도록 다음을
canonical JSON으로 직렬화해 contract SHA-256을 만든다.

- source manifest SHA-256
- teacher model metadata fingerprint
- max examples와 greedy decoding/max-new-token 설정
- min/max pixel budget
- image-SHA split 방식, seed, eval fraction
- local-only backend와 실행 설정

기존 raw cache가 있으면 provenance sidecar가 반드시 있어야 한다. 다음 중 하나라도
다르면 재개를 거부한다.

1. sidecar의 previous output SHA와 현재 raw-cache SHA
2. previous contract SHA와 새 contract SHA
3. 기존 row의 example key, image SHA, split, simulated flag, image byte

각 생성 뒤 전체 JSONL을 temporary sibling에 쓰고 atomic rename한다. 중단되면 마지막으로
완전히 기록된 row까지만 남는다. `resumed_examples`, `generated_examples`, partial/completed
상태가 provenance에 기록된다.

### Simulated flag

teacher backend 결과의 `simulated`는 `true/false`와 동등해 보이는 값이 아니라 **정확한
boolean type**이어야 한다. cache에는 `teacher_simulated`로 저장된다. 문자열 `"false"`,
정수 `0`, 누락 값은 invalid다. real audit와 student는 simulated row를 기본 거부한다.

## 4. 실제 단계 1 — 32B teacher cache

한 프로세스에서 teacher만 적재한다. 기본은 greedy decoding, 64 examples, 128 output
tokens, pixel 65,536–262,144, 60분, CUDA device 0, 최소 free VRAM 48 GiB다.

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python -m fmlab.vlm.response_distillation cache-teacher \
  --manifest "$FMLAB_DATA_ROOT/datasets/vlm/mixed/manifest.jsonl" \
  --output "$FMLAB_DATA_ROOT/generated/vlm/distillation/teacher-32b-raw.jsonl" \
  --teacher-model "$FMLAB_MODEL_ROOT/Qwen/Qwen3-VL-32B-Instruct-FP8" \
  --max-examples 64 \
  --max-new-tokens 128 \
  --max-minutes 60 \
  --split-seed 17 \
  --eval-fraction 0.25 \
  --cuda-device 0 \
  --min-free-vram-gib 48 \
  --min-pixels 65536 \
  --max-pixels 262144 \
  --execute
```

`--execute`는 실수로 heavy model을 올리지 않기 위한 명시적 gate다. 그 뒤에도 두 preflight가
있다.

1. **Resource preflight:** 선택한 GPU와 free VRAM을 확인하며 부족하면 model load 전에
   `blocked_resource`로 끝난다.
2. **One-request preflight:** 실제 model/backend로 한 요청을 처리해 FP8 runtime과 processor
   조합을 검증한 뒤에만 나머지 cache를 만든다.

2026-08-03 기록에서는 RTX PRO 6000 Blackwell, compute capability 12.0, 32B config의 FP8
E4M3/dynamic, quantizer availability를 확인했다. 그러나 48 GiB가 필요한 시점에 free VRAM이
약 6.38 GiB뿐이어서 model 적재 전에 `blocked_resource`로 종료됐다. **32B inference를
실행했다고 주장하지 않는다.**

cache 생성이 끝나면 이 프로세스를 완전히 종료하고 GPU 사용량을 확인한 뒤 다음 단계로
간다. 같은 Python process에서 teacher를 `del`한 뒤 student를 올리는 방식은 allocator,
library cache, 예외 복구 상태가 남을 수 있어 기본 계약이 아니다.

## 5. 실제 단계 2 — cache audit

teacher가 종료된 뒤 CPU에서 raw cache를 검사한다.

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python -m fmlab.vlm.response_distillation audit \
  --raw-cache "$FMLAB_DATA_ROOT/generated/vlm/distillation/teacher-32b-raw.jsonl" \
  --accepted-cache "$FMLAB_DATA_ROOT/generated/vlm/distillation/teacher-32b-accepted.jsonl" \
  --audit-json "$FMLAB_DATA_ROOT/generated/vlm/distillation/audit.json"
```

기본 rejection 조건:

- `teacher_simulated`가 boolean이 아니거나 true
- split이 `train/eval`이 아님
- image 누락, SHA 누락, 현재 image byte와 SHA 불일치
- 같은 task·image·question input 중복
- 빈 답, 길이 범위 위반
- refusal 또는 placeholder
- question에 reference answer가 노출됨
- train/eval image-SHA overlap

거부한 row는 버리지 않고 rejected JSONL과 reason count로 남긴다. teacher와 reference가
다른 것은 selection bias를 만들 수 있으므로 자동 rejection 조건이 아니다. exact match와
token F1을 quality audit로 보고할 뿐이다.

audit JSON은 다음을 고정한다.

- raw, accepted, rejected count와 acceptance rate
- rejection reason별 count
- train/eval example 및 image-group 수와 overlap
- task별 accepted count와 answer-length 통계
- raw-cache SHA, accepted-cache SHA, provenance path

`--allow-simulated`는 테스트 fixture를 수동 검사할 때만 사용한다. 실제 teacher cache에서
사용하지 않는다.

## 6. 실제 단계 3 — 8B student LoRA

새 프로세스에서 accepted cache만 읽는다. student는 `split=train` row만 사용하고 eval row는
학습에서 제외한다.

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python -m fmlab.vlm.response_distillation train-student \
  --cache "$FMLAB_DATA_ROOT/generated/vlm/distillation/teacher-32b-accepted.jsonl" \
  --audit-json "$FMLAB_DATA_ROOT/generated/vlm/distillation/audit.json" \
  --student-model "$FMLAB_MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct" \
  --output-dir "$FMLAB_DATA_ROOT/adapters/vlm/response-distilled-8b-lora" \
  --max-examples 64 \
  --max-steps 20 \
  --max-minutes 30 \
  --rank 8 \
  --cuda-device 0 \
  --min-free-vram-gib 24 \
  --min-pixels 65536 \
  --max-pixels 262144 \
  --execute
```

load 전에 student는 accepted-cache SHA와 row count가 audit JSON과 정확히 같은지 검증한다.
image byte SHA와 train/eval overlap도 다시 검사한다. simulated row가 하나라도 있으면
real-mode training을 거부한다.

학습 범위:

| 항목 | 값 |
|---|---|
| base | local Qwen3-VL-8B-Instruct |
| vision | frozen |
| LoRA | rank 8, alpha 16, dropout 0.05 |
| target | language `q_proj/k_proj/v_proj/o_proj` |
| sequence | 최대 2,048 |
| batch | 1 상당, gradient accumulation 4 |
| optimizer budget | 최대 20 step / 30분 |
| precision | BF16 |
| CUDA | device 0 명시, `device_map=auto` 금지 |
| storage | adapter-only |

student preflight도 24 GiB free VRAM gate를 통과하기 전 model을 적재하지 않는다. 기록된
호스트 시도는 약 6.4 GiB만 비어 있어 종료됐으므로 실제 8B optimizer step은 아직 없다.

## 7. Held-out 품질 평가

student training loss나 teacher agreement만으로 distillation 효과를 주장하지 않는다.
각 조건을 **별도 생성 job**으로 실행해 동일한 frozen holdout prediction JSONL을 만든 뒤
CPU evaluator로 비교한다.

각 row는 최소한 다음 필드를 가진다.

```json
{
  "id": "document-001-total",
  "image_sha256": "…",
  "task": "document_vqa",
  "answer_type": "number",
  "reference": "297.00",
  "prediction": "297"
}
```

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/evaluate_vlm_distillation.py \
  --base "$FMLAB_DATA_ROOT/predictions/vlm/base_8b.jsonl" \
  --gold "$FMLAB_DATA_ROOT/predictions/vlm/gold_lora.jsonl" \
  --distilled "$FMLAB_DATA_ROOT/predictions/vlm/distilled_lora.jsonl" \
  --teacher "$FMLAB_DATA_ROOT/predictions/vlm/teacher_32b.jsonl" \
  --train-manifest "$FMLAB_DATA_ROOT/generated/vlm/distillation/teacher-32b-accepted.jsonl" \
  --bootstrap-samples 2000 \
  --seed 17 \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/distillation-heldout"
```

감사된 cache를 입력하면 `--train-manifest`는 `split=train` 행만 train source로 해석한다.
evaluator는 id 집합, reference, task, answer type, image source를 조건 간 정렬하고 duplicate나
train/eval source overlap에서 실패한다. 출력:

- 조건별 exact match, ANLS, type-aware accuracy
- document/chart/grounding task slice
- distilled vs base, distilled vs gold의 paired bootstrap delta와 95% CI
- `P(delta > 0)`
- optional teacher-gap recovery
- base/gold/teacher/student의 방향별 실패 gallery
- prediction input SHA와 optional resource metrics

실제 한 seed output의 claim은 `heldout_comparison`이다. Experiment 승격 조건:

1. versioned manifest와 고정 image-SHA split
2. base/gold/distilled의 동일 prompt, decoding, pixel, example budget
3. 최소 seed 7, 17, 29
4. primary metric과 task slice의 paired interval
5. latency, peak VRAM, GPU-minutes, adapter bytes
6. teacher cache acceptance/rejection 및 failure cases 공개

## 8. 실패 복구와 감사 체크리스트

| 실패 | 보존되는 증거 | 복구 방법 |
|---|---|---|
| VRAM 부족 | preflight JSON, required/free bytes | 다른 GPU 작업 종료 후 같은 명령 재실행 |
| teacher 중단 | atomic raw cache, sidecar, resumed count | contract가 같을 때만 resume |
| cache 수동 변경 | output SHA mismatch | cache를 복원하거나 새 output 경로에서 재생성 |
| generation contract 변경 | contract SHA mismatch | 새 실험 디렉터리 사용 |
| invalid/simulated row | rejected JSONL과 reason | backend 문제 수정 후 teacher stage 재실행 |
| image 변경 | image SHA mismatch | dataset version을 새로 등록하고 cache 재생성 |
| audit/cache 불일치 | student load 전 실패 | accepted cache와 audit을 같은 run에서 다시 생성 |
| student OOM | failed metrics/preflight | pixels·sequence·steps를 등록된 새 cell로 축소 |

실패한 artifact를 성공 경로 위에 덮어쓰지 않는다. 설정을 바꾸면 별도 run id를 사용한다.

## 9. Artifact와 저장 공간

```text
distillation-run/
├── cache/
│   ├── teacher_raw.jsonl
│   ├── teacher_raw.provenance.json
│   ├── teacher_raw.preflight.json
│   ├── teacher_accepted.jsonl
│   ├── teacher_accepted.rejected.jsonl
│   └── audit.json
├── student/
│   ├── training_metrics.json
│   └── adapter files, real run only
├── loss_curve.svg
├── result.json
└── report.html
```

2026-08-03 최종 측정에서 전체 VLM artifact는 약 1.2 MB였다. 기존 32B FP8 model 약 34 GB와
8B model 약 17 GB는 공유 경로에서 읽고 복사하지 않는다. rank-8 `q/k/v/o`는 정확히
7,667,712 trainable parameters다. raw BF16은 약 14.6 MiB, PEFT가 FP32로 저장하면 약
29.3 MiB이므로 adapter당 약 15–31 MiB를 잡는다. logits와 optimizer state를 기본
보존하지 않으므로 이 트랙의 추가 용량은 1 GB보다 충분히 작다.

## 10. 설계 trade-off

장점:

- 32B와 8B의 VRAM co-residency가 없다.
- full logits 저장 공간이 없다.
- teacher output을 한 번 생성해 여러 student ablation에 재사용할 수 있다.
- cache와 audit이 hash-verified fingerprint로 연결되어 데이터 lineage를 검토할 수 있다.
- generation과 scoring이 분리되어 evaluator를 CPU에서 반복할 수 있다.

한계:

- teacher probability distribution의 dark knowledge를 전달하지 않는다.
- teacher의 체계적 OCR·reasoning 오류를 student가 모방할 수 있다.
- greedy response 한 개는 uncertainty를 표현하지 않는다.
- 작은 합성 canonical set은 실제 document distribution을 대표하지 않는다.

후속 저비용 실험은 gold-only vs audited teacher response를 먼저 비교하고, 효과가 있을 때만
raw vs audited, rank 4/8/16, low/medium pixel budget을 3 seed로 확장한다.

구현은 공식 [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL),
[Qwen3-VL-8B model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Transformers multimodal chat-template contract](https://huggingface.co/docs/transformers/chat_templating_multimodal),
[PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora)를 기준으로 한다.
