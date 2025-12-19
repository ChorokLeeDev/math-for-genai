# Chapter 5: Probability and Statistics

> **책 페이지**: 134-162
> **핵심 주제**: 확률 공간, 확률 변수, 분포 변환, Normalizing Flow, 다변량 가우시안, CLT
> **KAIST Challenge 연결**: Challenge 9 (Probability Transforms), Challenge 10 (Normalizing Flows)

---

## 📚 목차

1. [확률 vs 통계](#1-확률-vs-통계)
2. [확률 공간과 확률 변수](#2-확률-공간과-확률-변수)
3. [확률 분포의 변환](#3-확률-분포의-변환)
4. [Normalizing Flow 입문](#4-normalizing-flow-입문)
5. [다변량 확률 변수](#5-다변량-확률-변수)
6. [Central Limit Theorem과 극단값](#6-central-limit-theorem과-극단값)
7. [Notebooks 가이드](#7-notebooks-가이드)
8. [Generative AI에서의 응용](#8-generative-ai에서의-응용)

---

## 1. 확률 vs 통계

### 두 분야의 차이

> **책 원문 (p.134):**
> "Probability theory provides a mathematical framework for modeling uncertainty... Statistics is concerned with analyzing data."

```
확률론 (Probability):
    "분포가 주어졌을 때, 샘플은 어떻게 생겼을까?"
    P(X) → samples

통계학 (Statistics):
    "샘플이 주어졌을 때, 분포는 어떻게 생겼을까?"
    samples → P(X) 추정
```

### AI에서 왜 확률이 중요한가?

> **책 원문 (p.134):**
> "In AI, probability theory forms the foundation of probabilistic models such as Bayesian networks, variational autoencoders, normalizing flows, and diffusion models."

| 불확실성 유형 | 예시 | 해결 도구 |
|--------------|------|----------|
| **Model Uncertainty** | 모델의 예측이 얼마나 확실한가? | 베이지안 추론, VAE |
| **Data Uncertainty** | 데이터에 노이즈가 있다 | MLE, KDE |
| **Computational Uncertainty** | 정확한 계산이 불가능하다 | Monte Carlo, 변분 추론 |

---

## 2. 확률 공간과 확률 변수

### 확률 공간의 세 요소

> **책 원문 (p.137):**
> "A probability space formalizes randomness through a triple (Ω, F, P)"

```
(Ω, F, P):

Ω = 표본 공간 (Sample Space)
    "일어날 수 있는 모든 결과의 집합"
    예: 동전 던지기 → Ω = {Head, Tail}

F = σ-대수 (σ-algebra)
    "우리가 관심 있는 사건들의 집합"
    예: F = {∅, {Head}, {Tail}, Ω}

P = 확률 측도 (Probability Measure)
    "각 사건에 0과 1 사이 숫자 부여"
    예: P(Head) = 0.5, P(Tail) = 0.5
```

### 확률 변수 (Random Variable)

**정의**: 결과를 숫자로 바꾸는 함수 $X: \Omega \to \mathbb{R}$

```
동전 던지기:
    X(Head) = 1
    X(Tail) = 0

주사위:
    X(ω) = ω  (1, 2, 3, 4, 5, 6)
```

### 이산 vs 연속

| 타입 | 확률 표현 | 예시 |
|------|----------|------|
| **이산 (Discrete)** | PMF: $P_X(x) = P(X = x)$ | 동전, 주사위, 단어 |
| **연속 (Continuous)** | PDF: $p_X(x)$, $P(a \leq X \leq b) = \int_a^b p_X(x)dx$ | 키, 온도, 노이즈 |

### 기댓값과 분산

$$\mathbb{E}[X] = \sum_x x \cdot P(x) \quad \text{or} \quad \int x \cdot p(x) dx$$

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**직관**:
- 기댓값 = "평균적으로 어떤 값을 기대할 수 있나?"
- 분산 = "평균에서 얼마나 퍼져 있나?"

---

## 3. 확률 분포의 변환

### Change of Variables (변수 변환)

**핵심 질문**: $X$의 분포를 알 때, $Y = g(X)$의 분포는?

> **책 원문 (p.143):**
> "If f is differentiable and possibly not one-to-one, the change-of-variables formula in one dimension reads:"

$$p_Y(y) = \sum_{\text{pre-images } x: g(x)=y} p_X(x) \left| \frac{dx}{dy} \right|$$

### 예시 1: 아핀 변환

$Y = aX + b$, $X \sim \mathcal{N}(0, 1)$

```
역함수: x = (y - b) / a
야코비안: |dx/dy| = 1/|a|

결과:
p_Y(y) = p_X((y-b)/a) × (1/|a|)
       = N(y; b, a²)

직관: 평균이 b로 이동, 분산이 a²배
```

### 예시 2: 제곱 변환

$Y = X^2$, $X \sim \mathcal{N}(0, 1)$

```
역함수: x = ±√y (두 개!)
야코비안: |dx/dy| = 1/(2√y)

결과:
p_Y(y) = p_X(√y) × 1/(2√y) + p_X(-√y) × 1/(2√y)
       = (1/√(2πy)) × e^(-y/2)

이것이 χ² 분포 (자유도 1)!
```

### Jacobian의 직관

```
작은 구간 [x, x+dx]가 [y, y+dy]로 변환될 때:

dx 길이 → dy 길이

확률은 보존되어야 함:
    p_X(x) dx = p_Y(y) dy

따라서:
    p_Y(y) = p_X(x) × |dx/dy|
            = p_X(x) / |dy/dx|
```

**다변량으로 확장**:

$$p_Y(\mathbf{y}) = p_X(g^{-1}(\mathbf{y})) \times \left| \det \frac{\partial g^{-1}}{\partial \mathbf{y}} \right|$$

---

## 4. Normalizing Flow 입문

### 핵심 아이디어

> **책 원문 (p.142):**
> "Modern generative models frequently construct complex target distributions by transforming simple base distributions through invertible maps (flows)."

```
Normalizing Flow의 목표:

간단한 분포 z ~ N(0, I)
        ↓
    역변환 가능한 f_θ
        ↓
복잡한 분포 x = f_θ(z) ~ p_data
```

### 왜 "Normalizing" Flow인가?

```
두 가지 방향:

생성 (Generation):
    z ~ N(0,1) → x = f(z) → 복잡한 데이터
    "정규분포를 데이터로 변환"

정규화 (Normalization):
    x ~ p_data → z = f⁻¹(x) → N(0,1)
    "데이터를 정규분포로 변환"
```

### 학습 목표

**Likelihood 최대화**:

$$\log p_X(x) = \log p_Z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|$$

```
두 항의 의미:

1. log p_Z(f⁻¹(x)):
   "x를 z로 변환했을 때, z가 표준정규에서 얼마나 그럴듯한가?"

2. log |det Jacobian|:
   "변환이 공간을 얼마나 늘리거나 줄이나?"
   (밀도 보존을 위한 보정)
```

### 1D Normalizing Flow 예시

노트북 `Normalizing-Flow-1D.ipynb`에서:

```python
# 목표: Laplace 분포를 Gaussian으로 변환

# 학습할 역변환 g_θ
# x ~ Laplace → z = g_θ(x) ~ N(0,1)

# Loss = -log p_Z(g_θ(x)) - log |dg_θ/dx|
```

---

## 5. 다변량 확률 변수

### 결합 분포 (Joint Distribution)

$$p(x_1, x_2, \ldots, x_n) = p(\mathbf{x})$$

**Chain Rule로 분해**:

$$p(x_1, \ldots, x_n) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1,x_2) \cdots$$

> 이것이 **Auto-regressive 모델**의 수학적 기초!

### 독립성

```
X, Y가 독립:
    p(X, Y) = p(X) × p(Y)

조건부 독립:
    p(X, Y | Z) = p(X|Z) × p(Y|Z)
    "Z를 알면, X와 Y는 서로 무관"
```

### 다변량 가우시안 (가장 중요!)

> **책 원문 (p.149):**
> "The multivariate Gaussian is perhaps the most important distribution in AI."

$$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

| 파라미터 | 의미 | 차원 |
|---------|------|------|
| $\boldsymbol{\mu}$ | 평균 벡터 (분포의 중심) | $d \times 1$ |
| $\boldsymbol{\Sigma}$ | 공분산 행렬 (퍼짐과 방향) | $d \times d$ |

### 공분산 행렬의 기하학적 의미

```
Σ = I (단위행렬):
    원형 분포, 축 정렬

Σ = [[4, 0], [0, 1]]:
    타원형, x축으로 더 퍼짐

Σ = [[2, 1], [1, 2]]:
    타원형, 45도 기울어짐
```

**고유값 분해와 연결** (Ch.1 복습):

$$\Sigma = V \Lambda V^\top$$

- $V$의 열: 타원의 주축 방향
- $\Lambda$의 대각 원소: 각 방향의 분산

### Whitening = 타원 → 원

$$\mathbf{z} = \Sigma^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$$

- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$
- 모든 방향에서 분산 1, 상관관계 0

> Diffusion, VAE의 핵심!

---

## 6. Central Limit Theorem과 극단값

### Central Limit Theorem (CLT)

> **책 원문 (p.155):**
> "The CLT explains the emergence of Gaussian structure in sums of random variables."

$$\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

**직관**:
```
어떤 분포든 (유한한 평균과 분산만 있으면):
    많이 더하면 → 정규분포로 수렴!

예:
    주사위 1개: 균등 분포 (1~6)
    주사위 30개의 합: 거의 정규분포!
```

**AI에서의 의미**:
- 왜 Gaussian 노이즈를 많이 쓰는가? → CLT!
- SGD 노이즈 → 많은 미니배치의 평균 → 대략 Gaussian

### 희귀 사건과 Poisson 분포

```
n → ∞, p → 0, np → λ (일정)

Binomial(n, p) → Poisson(λ)

예: 하루에 웹사이트 방문자 수
    - 전체 인구: 매우 큼 (n → ∞)
    - 개인이 방문할 확률: 매우 작음 (p → 0)
    - 평균 방문자 수: λ (상수)
```

### 극단값 분포 (Extreme Value)

```
"평균"이 아닌 "최대값"의 분포는?

max(X₁, X₂, ..., Xₙ)의 분포

→ 세 가지 극단값 분포로 수렴:
   Gumbel, Fréchet, Weibull

AI 응용:
- 최악의 경우 분석
- 이상치 탐지
- 강건성 평가
```

---

## 7. Notebooks 가이드

### Statistics/ 폴더

| 노트북 | 뭘 배우나? | 핵심 실습 |
|--------|-----------|----------|
| `Empirical-Distributions-1D.ipynb` | 경험적 분포 | 히스토그램, ECDF |
| `Transformations-1D.ipynb` | 변수 변환 | $Y = X^2$, $Y = \tanh(X)$ |
| `Inverse-CDF.ipynb` | 역 CDF 샘플링 | 균등분포에서 임의 분포 생성 |
| `Normalizing-Flow-1D.ipynb` | 1D Flow | 역변환 학습 |
| `Empirical-Multivariate.ipynb` | 다변량 경험 분포 | 2D 시각화 |
| `Aggregate-Rare-Events.ipynb` | CLT, Poisson | 희귀 사건 |

### 꼭 해볼 실험들

**1. 경험적 분포의 수렴**
```python
# 샘플 수를 늘리면서 히스토그램 관찰
# N = 10, 100, 1000, 10000
# 어떻게 진짜 분포에 수렴하나?
```

**2. 변수 변환 시각화**
```python
# X ~ N(0,1)일 때
# Y = exp(X) → 로그정규분포
# Y = X² → 카이제곱
# 직접 확인해보기
```

**3. Normalizing Flow 학습**
```python
# 타겟: 쌍봉 분포 (Gaussian mixture)
# 베이스: N(0,1)
# 얼마나 잘 학습되나?
```

---

## 8. Generative AI에서의 응용

### VAE의 확률론적 해석

```
인코더: q_φ(z|x)
    "x가 주어졌을 때, z의 분포는?"
    보통 z|x ~ N(μ_φ(x), σ²_φ(x))

디코더: p_θ(x|z)
    "z가 주어졌을 때, x의 분포는?"

Prior: p(z) = N(0, I)
    "잠재 공간의 사전 분포"

학습 목표:
    ELBO = E[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

### Diffusion의 Forward Process

```
Forward (노이즈 추가):
    x₀ ~ p_data
    x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε,  ε ~ N(0,I)

t → ∞:
    x_T ~ N(0, I)  (거의 순수 노이즈)

이게 왜 Gaussian?
→ CLT! 노이즈가 누적되면서 정규분포로 수렴
```

### Normalizing Flow 생성 모델

```
Real NVP, Glow 등:

z ~ N(0, I)
    ↓ Affine Coupling Layer
    ↓ Affine Coupling Layer
    ↓ ...
x = f(z)

장점:
- 정확한 likelihood 계산 가능
- 학습과 샘플링 모두 효율적

단점:
- 표현력의 한계 (역변환 가능해야 함)
```

### Reparameterization Trick

VAE 학습의 핵심 기법:

```
문제:
    z ~ q_φ(z|x) = N(μ, σ²)에서 샘플링
    → gradient가 흐르지 않음!

해결 (Reparameterization):
    ε ~ N(0, 1)  (파라미터 무관)
    z = μ + σ × ε  (deterministic 변환!)
    → gradient가 μ, σ로 흐름!
```

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **확률 공간 (Ω, F, P)**
   - 모든 확률의 수학적 기초

2. **변수 변환 공식**
   - $p_Y(y) = p_X(g^{-1}(y)) \cdot |J^{-1}|$
   - Normalizing Flow의 기반

3. **다변량 가우시안**
   - $\mathcal{N}(\mu, \Sigma)$
   - VAE, Diffusion의 핵심 분포

4. **CLT (중심극한정리)**
   - 많이 더하면 정규분포
   - 왜 Gaussian이 어디에나 나타나는가

5. **Whitening**
   - 타원 → 원
   - Diffusion, Flow의 전처리

---

## 🔗 다른 챕터와의 연결

| 연결 | 설명 |
|------|------|
| **Ch.1 → Ch.5** | 공분산 행렬의 고유값 분해 |
| **Ch.5 → Ch.6** | 엔트로피, KL divergence |
| **Ch.5 → Ch.7** | 확률 분포 → 확률 과정 |
| **Ch.5 → Ch.8** | VAE, 에너지 기반 모델 |
| **Ch.5 → Ch.9** | Diffusion의 확률론적 기초 |

---

*이 문서는 Mathematics of Generative AI Book Chapter 5의 학습 가이드입니다.*
