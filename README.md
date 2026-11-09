# ESAA YB 1조 · 미니 프로젝트 2 — 와인 품질 분류

ESAA 26-2 YB 1조의 미니 프로젝트(분류) 최종 제출 레포임.
와인 성분과 종류를 바탕으로 quality 등급을 분류함.

- 팀원: 김태희, 오유빈, 정수연, 최윤서
- 대회: [데이콘 와인 품질 분류](https://dacon.io/competitions/open/235610/overview/description)
- 평가 지표: Accuracy
- 데이터: `train.csv`, `test.csv`, `sample_submission.csv` ([데이터 페이지](https://dacon.io/competitions/open/235610/data))

## 진행 흐름

1. 데이터 탐색 및 시각화
2. 전처리
3. 베이스라인 모델 (RandomForest)
4. ExtraTrees 모델 + 교차검증
5. 데이콘 제출 및 점수 기록

## 1. 데이터 탐색 (EDA)

- train 5,497행 · test 1,000행
- 변수: `index`, `quality`(타깃), 와인 성분 11개, `type`(red/white)
- 결측치 없음
- 변수 간 상관관계 히트맵, 변수별 분포, quality 등급별 변수 평균(barplot) 확인

## 2. 전처리

- `type`(white/red)을 LabelEncoder로 0/1 변환
- `index`, `quality`를 제거해 입력 변수 12개 구성
  - train_x: (5497, 12), test_x: (1000, 12)

## 3. 베이스라인 모델

- `RandomForestClassifier()` 기본 설정으로 학습 후 test 예측
- 제출 파일: `baseline.csv`

## 4. ExtraTrees 모델

- `ExtraTreesClassifier(n_estimators=1000, random_state=42)`
- StratifiedKFold 5-fold 교차검증으로 성능 확인
- **CV 정확도: 0.6802**
- 전체 train으로 다시 학습해 test 예측
- 제출 파일: `extratrees_1000.csv`

## 결과

| 모델 | CV Accuracy | Public 점수 |
|---|---|---|
| RandomForest (기본) | - |  |
| ExtraTrees (트리 1000개) | 0.6802 |  |

## 팀원별 작업

### 태희

-

### 유빈

-

### 수연

-

### 윤서

-

## 실행 방법

코랩에서 노트북을 열고 데이터 경로를 본인 드라이브 경로로 바꿔서 실행함.

```python
train = pd.read_csv("/content/drive/MyDrive/ESAA/ESAA미니프로젝트2/train.csv")
```

필요 라이브러리: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
