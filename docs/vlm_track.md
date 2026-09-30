# VLM 연구 트랙

이 트랙은 Qwen3-VL을 단순히 한 번 호출하는 데서 끝나지 않는다. 데이터 계약, 시각 token
예산, assistant-only supervision, image-level leakage 방지, LoRA target 검증, teacher cache
무결성, paired held-out 평가까지 작은 비용으로 관찰할 수 있는 연구 시스템을 만든다.

현재 집중하는 두 프로젝트는 다음과 같다.

1. vision encoder를 동결한 **Qwen3-VL-8B gold-label LoRA**
2. FP8 32B teacher 응답으로 학습하는 **Qwen3-VL-8B response-distilled LoRA**

최신 실제 GPU 증거는 [2026-08-04 VLM GPU smoke 보고서](results_2026-08-04-vlm-gpu-smoke.md),
이전 multimodal 계약은 [2026-08-03 VLM 보고서](results_2026-08-03-vlm.md), 자세한 증류
계약은 [response-distillation 문서](vlm_response_distillation.md)에 있다.

## 주장 수준

| 구성요소 | 현재 수준 | 증거 | 지원하지 않는 주장 |
|---|---|---|---|
| 합성 데이터·평가 | **Wiring** | 문서·차트·grounding·video의 deterministic pipeline과 HTML | Qwen3-VL 품질 |
| 실제 local processor | **real_processor_wiring** | 실제 이미지 2개, Qwen3VLProcessor, batch와 label mask | model forward, gradient, 품질 |
| 실제 8B 문서 추론 | **Smoke** | BF16 단일 이미지에서 `297.00` exact match | 데이터셋 평균 성능 |
| 8B LoRA | **Smoke — actual GPU** | 실제 local BF16 모델, vision-frozen text `q/k/v/o` LoRA, optimizer 2 step, 공개 bundle | 수렴·일반화·품질 향상·속도 향상 |
| 32B→8B 증류 | **Wiring + failed actual teacher attempt** | simulated cache·audit·student·report와 실제 32B model-load/FP8-kernel 실패 기록 | teacher generation 성공 또는 student 향상 |
| 자원·호환성 preflight | **Resource/failure evidence** | free-VRAM gate 통과, 32B 약 32.6 GiB 할당, 첫 generation의 `Unknown recipe` 실패 | inference 성공 또는 kernel 호환성 |

mock loss는 학습 경로를 검사하기 위한 의도적인 surrogate 값이다. `simulated: true` 또는
`quality_evaluated: false`가 붙은 수치를 모델 결과표에 옮기지 않는다.

## 구현 지도

| 파일 | 책임 |
|---|---|
| [`synthetic.py`](../src/fmlab/vlm/synthetic.py) | seed 고정 합성 문서·차트·정답 manifest 생성 |
| [`manifest.py`](../src/fmlab/vlm/manifest.py) | 문서·차트·grounding manifest를 canonical 파일로 결합 |
| [`lora_data.py`](../src/fmlab/vlm/lora_data.py) | schema 정규화, QA 확장, image-SHA split, chat content, assistant-only collator |
| [`lora_audit.py`](../src/fmlab/vlm/lora_audit.py) | LoRA target module과 vision-freeze 범위 검사 |
| [`lora_e2e.py`](../src/fmlab/vlm/lora_e2e.py) | 실제/tiny backend, optimizer loop, before/after 평가, provenance, JSON·HTML |
| [`response_distillation.py`](../src/fmlab/vlm/response_distillation.py) | teacher cache, resume, audit, student LoRA, resource preflight |
| [`distillation_evaluation.py`](../src/fmlab/vlm/distillation_evaluation.py) | 조건별 prediction의 CPU paired held-out 평가와 failure gallery |
| [`research.py`](../src/fmlab/vlm/research.py) | ANLS·type-aware score, bootstrap, fingerprint, provenance 공통 도구 |
| [`inference.py`](../src/fmlab/vlm/inference.py) | local-only Qwen3-VL 이미지·video 추론 |
| [`document_compare.py`](../src/fmlab/vlm/document_compare.py) | OCR→text QA와 direct VLM 비교 |
| [`chart_eval.py`](../src/fmlab/vlm/chart_eval.py) | chart operation slice 평가 |
| [`grounding.py`](../src/fmlab/vlm/grounding.py) | hallucination, bbox parse, IoU와 overlay |
| [`video.py`](../src/fmlab/vlm/video.py) | 명시적 frame sampling과 visual-cost 계획 |

## 1. 환경과 데이터

호스트에서 명령을 실행할 때마다 환경을 명시한다.

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
```

경로 계약은 다음과 같다.

```text
FMLAB_DATA_ROOT=${FMLAB_DATA_ROOT}
FMLAB_MODEL_ROOT=${FMLAB_MODEL_ROOT}
HF_HOME=${FMLAB_DATA_ROOT}/cache/huggingface
TMPDIR=${FMLAB_DATA_ROOT}/tmp
```

모델은 `FMLAB_MODEL_ROOT`에서 `local_files_only=True`로 읽는다. weight를 repository나
`/data`에 복사하지 않는다.

### Offline visual demo

모델 없이 데이터·평가·시각화 경로를 먼저 확인한다.

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python - <<'PY'
from pathlib import Path
from fmlab.vlm.demo import run_offline_demo

result = run_offline_demo(
    Path("${FMLAB_DATA_ROOT}/artifacts/vlm/offline-demo"),
    sample_count=4,
    seed=7,
)
print(result.to_dict())
PY
```

이 모드는 다음을 만든다.

- 합성 invoice·receipt·bar chart·shape image와 원본 label
- OCR noise가 downstream document QA를 망가뜨리는 비교
- lookup, argmax/min, sum, difference별 chart QA
- present/absent grounding과 hallucination rate
- 60초 video에서 12개 frame을 고르는 timeline

offline 정확도 1.0은 정답을 되돌리는 deterministic simulator 결과다.

### Canonical manifest

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/prepare_vlm_manifests.py
```

원본 manifest에는 한 이미지 아래 여러 QA가 중첩되거나 grounding 질문이 한 row씩 있는 등
schema가 다르다. loader는 모두 `{id, task, image_path, question, answer, answer_type}` 형태의
atomic example로 정규화한다. 현재 canonical set은 60개다.

| Task | QA 수 |
|---|---:|
| Document VQA | 24 |
| Chart QA | 20 |
| Grounding | 16 |

분할 key는 row id나 path 문자열이 아니라 실제 image byte의 SHA-256이다. 같은 픽셀을 다른
경로로 가리켜도 train/eval 양쪽으로 나뉘지 않는다.

## 2. 실제 processor 배선 검증

model weight를 적재하기 전에 실제 local Qwen3VLProcessor와 image processor, chat template,
collator를 확인한다.

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/check_vlm_processor.py \
  --model-path "$FMLAB_MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct" \
  --manifest "$FMLAB_DATA_ROOT/datasets/vlm/mixed/manifest.jsonl" \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/qwen3-vl-real-processor-check" \
  --batch-size 2 \
  --min-pixels 65536 \
  --max-pixels 401408
```

측정값:

| 항목 | 값 |
|---|---|
| examples | 2 |
| `input_ids` | `[2, 403]` |
| `pixel_values` | `[2992, 1536]` |
| `image_grid_thw` | 두 행 모두 `[1, 44, 34]` |
| merged visual tokens | 각 374 |
| supervised assistant tokens | 13 / 12 |
| supervised padding violations | 0 |
| duration | 0.215 s |

Qwen3-VL local chat template는 assistant mask를 직접 반환하지 않았다. 따라서 collator는
prompt-only conversation에 generation prompt를 붙인 token sequence와 complete conversation을
각각 처리한 뒤, prompt가 complete의 정확한 prefix인지 검사한다. 그 prefix 전체와 padding을
mask하고 assistant suffix만 학습한다. prefix가 맞지 않거나 sequence budget을 넘으면 조용히
자르지 않고 실패한다.

이 단계의 `claim_level`은 `real_processor_wiring`이다. processor와 실제 image를 사용했지만
model weight를 load하지 않았으므로 inference나 training smoke로 승격하지 않는다.

## 3. Track A — Qwen3-VL-8B LoRA

### 학습 계약

기본 설정은 [`qwen3_vl_8b_lora_e2e.yaml`](../configs/vlm/qwen3_vl_8b_lora_e2e.yaml)이다.

| 항목 | 기본값 | 이유 |
|---|---:|---|
| vision encoder | frozen | 작은 데이터에서 비용과 catastrophic drift 제한 |
| target | `q/k/v/o` | language attention만 적응하고 범위를 audit하기 쉬움 |
| rank / alpha | 8 / 16 | 7,667,712 trainable parameters의 작은 기준선 |
| train / eval budget | 최대 16 / 3 | task coverage를 지키는 bounded smoke |
| batch / accumulation | 1 / 4 | activation peak 제한 |
| steps | 2 | gradient path만 검증 |
| sequence | 1,024 | image+prompt+answer 전체 상한 |
| pixel | 65,536–401,408 | OCR 가능성과 visual-token 비용의 절충 |
| CUDA | device 0 명시 | training 중 자동 model sharding 금지 |
| free VRAM | 최소 24 GiB | 다른 GPU workload를 침범하지 않는 fail-fast gate |
| adapter save | false | smoke에서 checkpoint 증가 방지 |

실제 Qwen3-VL language layer에 존재하지 않는 target, vision module에 잘못 붙은 adapter,
동결되지 않은 vision parameter, LoRA 외 trainable parameter가 있으면 model load 뒤 즉시
실패한다.

### Tiny end-to-end wiring

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/run_vlm_lora.py \
  --config configs/vlm/qwen3_vl_8b_lora_e2e.yaml \
  --backend mock \
  --max-steps 3 \
  --artifact-dir "$FMLAB_DATA_ROOT/artifacts/vlm/qwen3-vl-8b-lora-mock-task-aware"
```

최종 wiring run의 split과 budget은 다음과 같다.

| 구간 | QA | image groups | chart | document | grounding |
|---|---:|---:|---:|---:|---:|
| 전체 train split | 45 | 9 | 15 | 18 | 12 |
| 전체 eval split | 15 | 3 | 5 | 6 | 4 |
| 실제 train budget | 16 | subset | 6 | 5 | 5 |
| 실제 eval budget | 3 | subset | 1 | 1 | 1 |

train/eval image overlap과 missing task는 모두 0이다. tiny backend는 3 optimizer step과
before/after teacher-forced proxy, loss trace, parameter ratio, fingerprint, JSON, HTML을 만든다.
최종 tiny loss `5.525444746`은 **Qwen3-VL loss가 아니다.** 생성 품질 평가도 아니다.

### 실제 local smoke

GPU가 비어 있고 free VRAM이 24 GiB 이상일 때 실행한다.

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/run_vlm_lora.py \
  --config configs/vlm/qwen3_vl_8b_lora_e2e.yaml \
  --backend local \
  --max-steps 2 \
  --artifact-dir "$FMLAB_DATA_ROOT/artifacts/vlm/qwen3-vl-8b-lora-local-smoke"
```

adapter가 실제로 필요할 때만 `--save-adapter`를 붙인다. smoke 성공의 기준은 quality 향상이
아니라 model load, LoRA scope audit, assistant-only backward, optimizer step, before/after
generation, report 완료다.

2026-08-04 실제 실행에서는 24 GiB gate를 통과해 local BF16 `8,774,791,408`
parameter를 적재했다. vision encoder는 동결됐고 text attention의 `q/k/v/o`에 붙은
LoRA `7,667,712`개(`0.08738%`)만 trainable이었다. optimizer 2 step에서 assistant
supervised token 44개를 30.28 token/s로 처리했다.

| actual GPU smoke 항목 | 값 |
|---|---:|
| model load | 34.26 s |
| optimizer steps | 2 |
| training time | 1.45 s |
| total wall time | 36.75 s |
| training peak allocated | 17.45 GiB |
| grouped synthetic holdout | 3 examples / 3 image groups |
| exact match, before → after | 100% → 100% |
| paired delta, 95% CI | 0 / `[0,0]` |

holdout 기준선이 이미 완벽했으므로 품질 향상은 관찰할 수 없었다. before/after latency도
warm-cache 순서 효과와 분리되지 않아 속도 향상으로 해석하지 않는다. 이 실행은 실제
model load·LoRA scope·assistant-only backward·optimizer·memory path의 **Smoke** 증거다.
수렴·일반화·competitive quality 증거가 아니다.
[공개 result](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/result.json) ·
[HTML report](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/report.html) ·
[evidence manifest](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/evidence_manifest.json)

## 4. Track B — response distillation

teacher와 student를 동시에 GPU에 올리지 않고 `cache-teacher → process exit → audit →
train-student`로 실행한다. 기본 teacher는 local Qwen3-VL-32B-Instruct-FP8, student는 같은
8B와 rank-8 `q/k/v/o` LoRA다.

full mock wiring:

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python -m fmlab.vlm.response_distillation mock \
  --manifest "$FMLAB_DATA_ROOT/datasets/vlm/mixed/manifest.jsonl" \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/response-distillation-mock-full" \
  --max-examples 60 \
  --max-steps 5
```

이 실행은 simulated teacher 60개를 audit 후 모두 수용했다. train/eval은 45/15,
image groups는 9/3, overlap은 0이다. task totals는 document 24, chart 20, grounding 16이다.
mock loss `2.052309 → 1.115530`은 `quality_evaluated: false`로 기록된다.

2026-08-04 실제 teacher 시도에서는 GPU가 비어 48 GiB gate가 통과했고, 32B FP8 모델은
적재되어 약 32.6 GiB를 할당했다. 그러나 첫 generation에서 Transformers가 선택한
`kernels-community/finegrained-fp8` v1이 Blackwell runtime에서 `Unknown recipe`를 내며
실패했다. 최신 v4는 노출 API가 달라 현재 integration에 그대로 교체할 수 없었다.
teacher row는 0개이며 audit/student 실제 실행은 없었다. 그러므로 이 기록은 model-load와
compatibility failure 증거일 뿐 32B inference 성공이나 distillation 품질 증거가 아니다.
raw 실패 provenance에는 로컬 경로와 host 정보가 있어 공개 bundle으로 내지 않았다.

실제 세 단계 명령, resume 계약, simulated-target 거부, SHA pin은
[response-distillation 문서](vlm_response_distillation.md)를 따른다.

## 5. Held-out evaluator

학습 runner의 before/after 값만으로 두 트랙을 비교하지 않는다. base 8B, gold-LoRA,
distilled-LoRA, 선택적 32B teacher가 같은 frozen example에 생성한 prediction을 별도 파일로
저장한 뒤 CPU evaluator로 재채점한다.

prediction row 최소 schema:

```json
{"id":"doc-001","image_sha256":"…","task":"document_vqa","answer_type":"number","reference":"297.00","prediction":"297"}
```

```bash
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
evaluator는 다음 조건에서 fail closed한다.

- 세 필수 조건의 example id 집합이 다름
- reference, task, answer type, image source가 서로 다름
- duplicate id가 있음
- train/eval image SHA가 겹침
- prediction 또는 reference가 없음

출력은 exact match, ANLS, type-aware accuracy의 전체·task slice, paired bootstrap 95% CI,
`P(delta>0)`, teacher-gap recovery, 방향별 failure gallery다.

실제 prediction으로 한 번 실행해도 claim은 `heldout_comparison`이다. 다음을 모두 충족할 때만
**Experiment**로 승격한다.

1. 고정된 image-SHA split과 dataset fingerprint
2. 8B zero-shot, gold-label LoRA, distilled LoRA의 동일 decoding·pixel budget
3. 최소 seed 7, 17, 29
4. paired interval과 per-task 결과
5. adapter bytes, peak VRAM, wall time, GPU-minutes
6. failure gallery와 중단/실패 run 보존

## 6. 저비용 ablation

먼저 한 seed·20 step 이하로 screening하고 차이가 있는 cell만 3 seed로 승격한다.

| ID | 바꾸는 것 | 고정하는 것 | 질문 |
|---|---|---|---|
| B0 | 학습 없음 | split·decoding | 8B zero-shot 기준선 |
| G8 | gold, rank 8 `q/k/v/o` | vision frozen | 기본 LoRA |
| G8-qv | gold, rank 8 `q/v` | data·steps | target 절반의 효율 |
| G4 / G16 | rank 4 / 16 | target·steps | adapter capacity 대비 효과 |
| D8-raw | raw teacher response | rank 8 | audit 전 imitation |
| D8-audit | accepted cache | rank 8 | filtering의 추가 가치 |
| P8-low | 더 낮은 max pixels | best supervision | OCR 품질/visual-token 비용 곡선 |
| V8-last | 마지막 vision block 일부 | best setting | error slice가 정당화할 때만 시각 적응 |

primary metric은 문서 exact match·ANLS, 차트 tolerance-aware numeric/type accuracy,
grounding precision·recall·F1·hallucination rate다. 효율은 trainable parameter, adapter byte,
peak VRAM, wall time, GPU-minute, throughput으로 함께 보고한다.

## 7. 산출물과 저장 공간

LoRA run은 가능한 경우 다음을 남긴다.

```text
run/
├── result.json
├── provenance.json
├── training_trace.jsonl
├── predictions_before.jsonl
├── predictions_after.jsonl
└── report.html
```

distillation은 raw/accepted/rejected cache, cache provenance, audit JSON, student metrics,
loss SVG, result JSON, HTML을 추가한다. JSON이 원본이며 HTML은 파생 view다. 실패도 error와
resource 상태를 보존한다.

최종 측정 기준 소스는 약 2.5 MB, 외부 lab root는 약 653 MB, VLM artifact는
약 1.2 MB였다. 기존 8B 약 17 GB와 32B 약 34 GB model은 복사하지 않는다. rank-8
`q/k/v/o`는 7,667,712 trainable parameters이며 raw BF16은 약 14.6 MiB, PEFT FP32 저장은
약 29.3 MiB다. adapter당 약 15–31 MiB를 잡아도 optimizer state를 저장하지 않으면 두
트랙의 추가 공간은 1 GB보다 충분히 작다.

## 8. 공식 구현 계약과 제한

구현 기준:

- [Qwen3-VL 공식 repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Transformers multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora)

현재 제한:

- canonical 데이터는 작고 합성이므로 real document distribution을 대표하지 않는다.
- 실제 8B LoRA optimizer 2 step은 실행했지만 합성 holdout 3개 기준선이 이미 100%라
  품질 향상·수렴·일반화·속도 향상을 주장하지 않는다.
- 실제 32B teacher는 model load 뒤 FP8 kernel compatibility에서 실패해 cache row가 0개다.
  따라서 실제 teacher generation과 distillation student 결과는 없다.
- response distillation은 teacher sequence만 전달하므로 full-logit dark knowledge가 없다.
- filtering은 오류를 줄이는 감사 가능한 개입이지 teacher 정확성을 보장하지 않는다.
- 한 샘플 actual inference와 두 step smoke를 benchmark로 표현하지 않는다.
