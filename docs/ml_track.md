# 특징적인 ML 토이 프로젝트 트랙

이 트랙은 외부 다운로드 없이 합성 데이터만으로 실행된다. 각 실험은 같은 규칙을 따른다.

1. 고정된 seed로 데이터를 만든다.
2. 작은 모델 또는 고전 기준선을 학습한다.
3. 서로 다른 접근법을 같은 테스트 데이터에서 비교한다.
4. `result.json`에 설정과 수치 결과를 기록한다.
5. 사람이 결과를 눈으로 확인할 수 있는 PNG를 함께 저장한다.

전체 기본 실험은 다음처럼 실행한다.

```bash
cd /path/to/foundation-model-lab
python -m fmlab.ml.runner all \
  --config-dir configs/ml \
  --output-dir ${FMLAB_DATA_ROOT}/artifacts/ml
```

하나만 실행할 때는 다음 형식이다.

```bash
python -m fmlab.ml.runner diffusion \
  --config configs/ml/diffusion.yaml \
  --output-dir ${FMLAB_DATA_ROOT}/artifacts/ml/diffusion
```

`device: auto`는 CUDA가 있으면 GPU를, 없으면 CPU를 사용한다. 모든 기본 설정은 CPU에서도
학습 가능한 크기이며 모델이나 데이터를 내려받지 않는다.

## 1. Tiny DDPM 확산 모델

파일: `src/fmlab/ml/diffusion.py`

8×8 막대·십자·대각선 이미지를 만든 뒤, 정방향 과정에서 가우시안 노이즈를 단계적으로
섞는다. 작은 합성곱 신경망은 원본 이미지가 아니라 **추가된 노이즈**를 예측한다. 생성 시에는
무작위 노이즈에서 출발하여 예측한 노이즈를 여러 번 제거한다.

- `forward_process.png`: 시간이 갈수록 이미지 구조가 사라지는 정방향 과정
- `training_loss.png`: 노이즈 예측 MSE
- `samples.png`: 역확산으로 생성한 최종 이미지
- 핵심 수치: 초기/최종 noise MSE, 생성 샘플과 가장 가까운 훈련 샘플의 MSE

볼 점은 훈련 loss가 낮아지는 것만으로 샘플이 즉시 선명해지지는 않는다는 사실이다.
`timesteps`, `train_steps`, `hidden_channels`를 하나씩 바꾸면 계산량과 생성 품질의 관계를 볼 수
있다.

## 2. CLIP 방식 이미지–텍스트 대조학습

파일: `src/fmlab/ml/contrastive.py`

합성 심볼 이미지와 심볼을 가리키는 텍스트 토큰을 한 쌍으로 만든다. 이미지 인코더와 텍스트
인코더는 각각 벡터를 출력하고, 올바른 쌍의 코사인 유사도는 높이고 잘못된 쌍은 낮추는 대칭
contrastive loss로 학습한다.

- `similarity_matrix.png`: 밝은 대각선이 올바른 이미지–텍스트 정렬
- `embedding_space.png`: 이미지와 텍스트 벡터를 PCA로 2차원에 투영
- `training_loss.png`: 대칭 contrastive loss
- 핵심 수치: image→text 및 text→image Recall@1/3

이 실험은 실제 CLIP의 핵심 목적함수만 분리한다. 실제 캡션 토크나이저와 대형 이미지
인코더는 의도적으로 제외하여, 공유 임베딩 공간이 만들어지는 과정에 집중한다.

## 3. 이상 탐지: 통계 모델과 오토인코더

파일: `src/fmlab/ml/anomaly.py`

곡선 형태의 정상 데이터만 학습에 사용한다. Mahalanobis 거리는 데이터가 하나의 타원형 분포라는
가정 아래 거리를 계산하고, 오토인코더는 정상 데이터의 비선형 저차원 구조를 복원한다. 두 방식
모두 정상 훈련 점수의 분위수로 임계값을 정한다.

- `anomaly_maps.png`: 정답, Mahalanobis 점수, 재구성 오차를 같은 좌표에서 비교
- `score_distributions.png`: 정상/이상 점수 분포와 판정 임계값
- `autoencoder_loss.png`: 정상 데이터 재구성 학습 곡선
- 핵심 수치: AUROC, precision, recall, F1, threshold

`threshold_quantile`을 높이면 오탐은 줄고 미탐이 늘어난다. AUROC는 임계값과 무관한 순위
품질이고 F1은 선택한 임계값까지 포함한 품질이라는 차이도 확인할 수 있다.

## 4. 시계열 예측 비교

파일: `src/fmlab/ml/timeseries.py`

추세, 계절성, 고조파, 노이즈가 있는 시계열의 다음 한 점을 예측한다. 미래를 훈련 데이터에
섞지 않고 시간순으로 분할하며 네 방법을 비교한다.

- 직전 값을 그대로 쓰는 persistence
- 한 계절 전 값을 쓰는 seasonal naive
- 선형 ridge autoregression
- 작은 비선형 MLP

`forecast_comparison.png`에서 실제 미래와 네 예측을 겹쳐 보고, `error_bars.png`에서 RMSE와
MAE를 비교한다. 신경망이 단순 기준선을 이기지 못하는 경우도 정상적인 연구 결과다. 계절성이
강한 데이터에서는 특히 seasonal naive가 강력하다.

## 5. 그래프 노드 분류: MLP, GCN, GAT

파일: `src/fmlab/ml/graph.py`

세 커뮤니티가 있는 stochastic block model 그래프를 만들고, 각 클래스의 일부 노드에만 라벨을
공개한다. MLP는 노드 특성만 보고, GCN은 이웃 특성을 정규화 평균하며, GAT는 어떤 이웃을
참고할지 attention weight를 학습한다.

- `community_predictions.png`: 실제 커뮤니티와 GCN/GAT 예측
- `training_curves.png`: 세 모델의 loss와 validation accuracy
- `model_comparison.png`: 공개하지 않은 노드의 test accuracy
- 핵심 수치: 모델별 train/validation/test accuracy, edge 수, 공개 라벨 수

`between_class_edge_probability`를 올리면 이웃이 항상 같은 클래스라는 가정이 깨진다. 이 값을
바꾸는 실험은 graph homophily와 GNN 성능의 관계를 보여준다.

## 6. 불확실성과 확률 보정

파일: `src/fmlab/ml/calibration.py`

겹치는 3개 클래스 데이터를 분류한 뒤 logit을 의도적으로 키워 과신 상태를 만든다. 별도의
validation 데이터에서 온도 `T` 하나를 고르고, test logit을 `logit / T`로 바꾼다. 이 과정은
예측 클래스를 바꾸지 않고 확률의 크기만 조정한다.

- `reliability_diagram.png`: confidence와 실제 정답률의 일치 정도
- `confidence_histogram.png`: 보정 전후 최대 확률 분포
- `confidence_surface.png`: 결정 공간 전체에서 과신이 완화되는 모습
- 핵심 수치: accuracy, NLL, Brier score, ECE, 선택된 temperature

accuracy가 그대로인데 ECE와 NLL이 좋아지는 것이 핵심 결과다. 이는 정확도와 “내 확률을 믿을
수 있는가”가 서로 다른 문제임을 보여준다.

## 결과 읽는 순서

각 결과 디렉터리에서 먼저 PNG를 보고, 다음으로 `result.json`의 `metrics`를 확인한다.
`parameters`에는 재현에 필요한 전체 설정, `artifacts`에는 그림 목록, `notes`에는 해석 시 주의점이
기록된다. 두 설정을 비교할 때는 seed를 고정하고 관심 변수 하나만 바꾸는 것이 가장 이해하기
쉽다.

빠른 동작 확인용 테스트는 다음과 같다.

```bash
pytest -q tests/test_ml_smoke.py
```

테스트 설정은 코드 경로와 산출물 형식만 검증하기 위해 학습을 2~3회만 수행한다. 의미 있는
학습 곡선과 품질 비교에는 `configs/ml/*.yaml`의 기본 설정을 사용해야 한다.
