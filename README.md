# README - 특성 선택(Feature Selection) 기반 ML 모델 프로젝트

## 프로젝트 소개

이 프로젝트는 머신러닝에서 중요한 과정 중 하나인 **특성 선택(Feature Selection)** 을 활용하여  
불필요한 데이터를 제거하고 핵심 특성만 선택하여 분류 모델을 학습하는 프로그램입니다.

프로그램은 다음과 같은 흐름으로 동작합니다.

1. 가상 분류 데이터 생성
2. 분산 기반 특성 제거
3. 통계 기반 특성 선택
4. RandomForest 기반 중요 특성 선택
5. 최종 선택된 특성으로 모델 학습
6. Gradio UI를 통한 예측 인터페이스 제공

---

## 사용 기술

- Python
- NumPy
- Pandas
- Scikit-learn
- Gradio

---

## 설치 방법

### 1. 필요한 라이브러리 설치

```bash
pip install numpy pandas scikit-learn gradio
```

---

## 실행 방법

```bash
python main.py
```

실행 후 브라우저에서 아래 주소 접속:

```bash
http://127.0.0.1:7860
```

---

## 프로젝트 구조

```bash
project/
│
├── main.py
└── README.md
```

---

## 코드 설명

### 1. 데이터 생성

```python
make_classification()
```

Scikit-learn의 가상 데이터 생성 함수를 사용하여  
20개의 특성을 가진 분류 데이터를 생성합니다.

---

### 2. VarianceThreshold

```python
VarianceThreshold
```

분산이 거의 없는 특성을 제거합니다.

예:
- 모든 값이 거의 동일한 특성 제거

---

### 3. SelectKBest

```python
SelectKBest(f_classif)
```

ANOVA F-value를 사용하여  
가장 중요한 상위 10개 특성을 선택합니다.

---

### 4. SelectFromModel

```python
SelectFromModel(RandomForestClassifier)
```

RandomForest 모델의 feature importance를 기반으로  
핵심 특성을 최종 선택합니다.

---

### 5. 모델 학습

```python
RandomForestClassifier
```

최종 선택된 특성으로 분류 모델을 학습합니다.

---

### 6. 정확도 평가

```python
accuracy_score
```
테스트 데이터에 대한 예측 정확도를 계산합니다.

---

### 7. Gradio 인터페이스

```python
gr.Interface
```
사용자가 직접 값을 입력하여  
실시간으로 클래스 예측 결과를 확인할 수 있습니다.

---

## 실행 화면 기능

### 입력 기능

- 선택된 핵심 특성 값 입력

### 출력 기능
- 클래스 예측 결과
- 클래스별 확률 출력

예시:
```text
예측 결과: 클래스 1
확률: 클래스 0 = 0.12, 클래스 1 = 0.88
```

---

## 머신러닝 흐름
```text
데이터 생성
   ↓
분산 기반 특성 제거
   ↓
통계 기반 특성 선택
   ↓
모델 기반 특성 선택
   ↓
모델 학습
   ↓
예측 및 정확도 평가
   ↓
Gradio UI 제공
```

---

## 프로젝트 목적
이 프로젝트의 목적은 다음과 같습니다.

- Feature Selection 개념 이해
- 데이터 차원 축소 학습
- 머신러닝 모델 성능 향상 이해
- Scikit-learn 활용 능력 향상
- Gradio 기반 인터페이스 구현
---

## 기대 효과
- 불필요한 특성을 제거하여 학습 속도 향상
- 중요한 데이터만 사용하여 모델 성능 개선
- 사용자 친화적인 ML 인터페이스 구현 가능

---
## 작성자
- BigData 학과 머신러닝 실습 과제
- Python 기반 Feature Selection 프로젝트