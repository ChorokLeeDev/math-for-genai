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

### 핵심 해결 도구 간단 소개

**① 베이지안 추론 (Bayesian Inference)**
> "데이터를 보고 믿음(확률)을 업데이트"

$$P(\theta|\text{Data}) \propto P(\text{Data}|\theta) \times P(\theta)$$

| 항 | 이름 | 의미 |
|---|---|---|
| $P(\theta \vert \text{Data})$ | 사후분포 | 데이터를 본 후의 믿음 |
| $P(\text{Data} \vert \theta)$ | 우도 | 파라미터가 데이터를 얼마나 잘 설명하나 |
| $P(\theta)$ | 사전분포 | 데이터를 보기 전의 믿음 |

- 점 추정이 아닌 분포로 불확실성 표현
- VAE의 잠재 변수 $z$의 사후분포 $q(z|x)$ 추론

**② MLE (Maximum Likelihood Estimation)**
> "데이터가 가장 그럴듯하게 나오는 파라미터 찾기"

$$\theta^* = \arg\max_\theta P(\text{Data}|\theta) = \arg\max_\theta \prod_i p(x_i|\theta)$$

- 빈도주의적 접근, 점 추정
- 신경망 학습의 손실함수: $-\log P(\text{Data}|\theta)$

**③ KDE (Kernel Density Estimation)**
> "샘플들 주변에 커널 함수를 놓고 합산하여 밀도 추정"

$$\hat{p}(x) = \frac{1}{n} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$

- $K$: 커널 함수 (보통 Gaussian)
- $h$: 대역폭 (smoothing 정도)
- 비모수적 밀도 추정의 기본

**커널 함수란?** 각 데이터 포인트 주변에 "언덕"을 쌓는 함수

```
       ▲
      /│\
     / │ \      ← 각 데이터 포인트가 만드는 언덕
    /  │  \
───●───●───●───  ← 데이터 포인트들
   x₁  x₂  x₃

모든 언덕을 합치면 → 부드러운 밀도 추정

대표적 커널:
• Gaussian: K(u) = (1/√2π) exp(-u²/2)  ← 가장 많이 사용
• Uniform:  K(u) = 1/2  (|u| ≤ 1)
• Epanechnikov: K(u) = 3/4(1-u²)  (|u| ≤ 1)

대역폭 h의 효과:
  h 작음 → 뾰족한 언덕 → 세밀하지만 노이즈에 민감
  h 큼   → 완만한 언덕 → 부드럽지만 디테일 손실
```

**④ Monte Carlo 방법**
> "직접 계산이 어려운 적분/기댓값을 샘플링으로 근사"

$$\mathbb{E}[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \quad x_i \sim p(x)$$

- 고차원 적분의 유일한 실용적 해법
- MCMC, Importance Sampling 등 다양한 변형
- Diffusion의 stochastic sampling 기반

**⑤ 변분 추론 (Variational Inference)**
> "다루기 어려운 사후분포를 다루기 쉬운 분포로 근사"

$$q^*(z) = \arg\min_q D_{KL}(q(z) \| p(z|x)) \approx \arg\max_q \text{ELBO}$$

**KL divergence란?**

두 분포가 얼마나 다른지 측정하는 숫자

$$D_{KL}(P \| Q) = \text{"P를 Q로 근사할 때 정보 손실"}$$

- $D_{KL} = 0$ → 두 분포가 완전히 같음
- $D_{KL} > 0$ → 분포가 다름 (클수록 더 다름)
- → Chapter 6에서 자세히!

**특징:**
- 적분 문제 → 최적화 문제로 변환
- VAE의 핵심 학습 원리
- 계산 효율적 (MCMC보다 빠름)

**$x$, $z$가 뭔가?** VAE에서의 의미:

- $x$ = 관측 데이터 (예: 이미지)
- $z$ = 잠재 변수 (latent variable, 저차원 표현)

**VAE의 목표:**

$$\text{이미지 } x \xrightarrow{\text{인코더}} z \text{ (압축)} \xrightarrow{\text{디코더}} \hat{x} \text{ (복원)}$$

**문제:** $p(z|x)$ = "이미지 $x$가 주어졌을 때 $z$의 진짜 분포" → 계산 불가능! (적분이 intractable)

**해결:** $q(z|x) \approx p(z|x)$ → 신경망으로 $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$ 근사

**"신경망으로 근사"의 의미**: 분포의 파라미터를 출력

```
인코더 신경망:

    이미지 x (784차원)
         ↓
    ┌─────────────┐
    │  Conv layers │
    │  FC layers   │
    └──────┬──────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
  μ(x)          σ(x)
 (10차원)      (10차원)
```

$$q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$$

핵심: 신경망이 "분포 자체"를 출력하는 게 아니라, 분포를 정의하는 숫자들 $(\mu, \sigma)$을 출력!

**ELBO의 두 항 자세히:**

$$\text{ELBO} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{복원 항}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{정규화 항}}$$

**① 복원 항:** $\mathbb{E}[\log p(x|z)]$

"$z$에서 $x$를 얼마나 잘 복원하나?"

$$z \to \text{디코더} \to \hat{x}, \quad \log p(x|z) \approx -\|x - \hat{x}\|^2 \text{ (MSE loss)}$$

- 크면 → 복원 잘 됨
- 작으면 → 흐릿하거나 다른 이미지

**② 정규화 항:** $D_{KL}(q(z|x) \| p(z))$

"인코더 출력이 표준정규분포에 가까운가?"

$$q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x)) \quad \text{vs} \quad p(z) = \mathcal{N}(0, I)$$

- KL 작으면 → 잠재 공간이 정규분포처럼
- KL 크면 → 잠재 공간에 구멍 많음

**왜 잠재 공간이 정규분포여야 하는가?**

생성할 때: $z \sim \mathcal{N}(0,I)$ 샘플링 → 디코더 → 새 이미지

```
만약 잠재 공간이 정규분포가 아니라면?

    학습된 잠재 공간:

        ●●●          ●●●
           (빈 공간)          ← 데이터가 있는 곳만 점이 있음
        ●●            ●●●

    생성 시 z ~ N(0, I)에서 뽑으면:

        ○  ← 빈 공간에서 뽑힘!

        → 디코더가 본 적 없는 z
        → 이상한 이미지 출력 (노이즈, 흐릿함)
```

```
KL 정규화로 잠재 공간을 N(0, I)에 가깝게 만들면?

    학습된 잠재 공간:

        ●●●●●●●●●
        ●●●●●●●●●    ← 빈 공간 없이 고르게 분포
        ●●●●●●●●●

    생성 시 z ~ N(0, I)에서 뽑으면:

        ○  ← 어디를 뽑아도 데이터가 있던 영역!

        → 디코더가 학습한 영역 안
        → 의미 있는 이미지 출력!
```

**비유: 지도**
- 잠재 공간 = 지도, 디코더 = "좌표 → 이미지" 변환
- 정규화 안 하면: 지도에 빈 땅(바다)이 많음 → 랜덤 좌표가 바다에 빠짐
- 정규화 하면: 지도 전체가 육지로 빽빽함 → 어디를 찍어도 의미 있는 이미지

**Trade-off:**
- 복원만 중시 → 각 $x$마다 $z$ 암기 → 샘플링 안 됨
- 정규화만 중시 → 모든 $x$가 같은 $z$ → 복원 안 됨
- 균형 → 복원도 되고, 샘플링도 됨!

**MCMC vs 변분 추론**:

```
MCMC (Markov Chain Monte Carlo):
    • 샘플링 기반 → 정확하지만 느림
    • 수렴 보장, 하지만 언제 수렴했는지 모름

변분 추론:
    • 최적화 기반 → 빠르지만 근사
    • 손실함수(ELBO) 보고 수렴 확인 가능
    • 대규모 데이터에 적합 → 딥러닝과 궁합 좋음
```

**왜 "딥러닝과 궁합 좋음"인가?**

```
MCMC의 한계:
    전체 데이터를 한 번에 봐야 샘플 하나 생성
    데이터 100만개 → 샘플 1개에 100만개 전부 필요
    → 미니배치 불가, 느림!

변분 추론의 장점:
    ELBO를 SGD로 최적화 (미니배치 OK!)

    for batch in dataloader:      # 64개씩
        loss = -ELBO(batch)       # 손실 계산
        loss.backward()           # 자동미분, gradient 계산 (∂loss/∂θ)
        optimizer.step()          # SGD/Adam, θ ← θ - η·∇loss 로 파라미터 업데이트

    → 일반 신경망 학습이랑 완전히 같은 방식!
    → PyTorch/TensorFlow에서 바로 구현 가능
```

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

**야코비안이란?** 변환이 공간을 얼마나 늘이거나 줄이는지 나타내는 숫자

#### 1차원 예시: 고무줄 늘이기

```
원본 공간 (x):    |●●●●●●●●●●|     점 10개, 길이 10cm
                                    밀도: 1개/cm

2배로 늘임 (y):   |●  ●  ●  ●  ●  ●  ●  ●  ●  ●|   점 10개, 길이 20cm
                                    밀도: 0.5개/cm (절반으로 감소!)
```

**점의 개수는 보존**되지만, 공간이 늘어나면 **밀도는 줄어듭니다**.

#### 확률도 마찬가지

확률의 총합은 항상 1 (보존됨). 하지만 공간이 변하면 **밀도**가 변합니다.

```
변환 y = 2x (2배 늘이기) 라면:

  원본                    변환 후
   x                        y
   ↓                        ↓
|──────|                |────────────|
  길이 1                    길이 2

밀도(확률)가 절반으로!
```

#### 수식으로 표현하면

```
작은 구간 [x, x+dx]가 [y, y+dy]로 변환될 때:

    dx 길이 → dy 길이

확률 질량은 보존되어야 함:
    p_X(x) × dx = p_Y(y) × dy
    ─────────    ─────────
    원본 구간의    변환 후 구간의
    확률 질량      확률 질량

따라서:
    p_Y(y) = p_X(x) × |dx/dy|
                      ───────
                      야코비안!
```

#### 야코비안의 역할 정리

| dy/dx (변환율) | 의미 | 야코비안 \|dx/dy\| | 밀도 변화 |
|---------------|------|-------------------|----------|
| 2 | 2배 늘어남 | 1/2 | 절반으로 감소 |
| 0.5 | 절반으로 압축 | 2 | 2배로 증가 |
| 1 | 변화 없음 | 1 | 그대로 |

#### 핵심 직관

> **공간이 늘어나면 → 밀도가 줄고**
> **공간이 압축되면 → 밀도가 늘어난다**
> **야코비안이 이 보정을 해준다!**

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

### Auto-regressive 모델이란?

**Auto-regressive = "이전 것들을 보고 다음 것을 예측"**

```
문장 생성 예시 (GPT):

"나는" → "밥을" → "먹었다" → "."

P(문장) = P("나는")
        × P("밥을" | "나는")
        × P("먹었다" | "나는", "밥을")
        × P("." | "나는", "밥을", "먹었다")
```

이게 바로 위의 Chain Rule 분해와 **똑같은 구조**입니다!

```
Chain Rule:     p(x₁) × p(x₂|x₁) × p(x₃|x₁,x₂) × ...
                  ↓        ↓            ↓
GPT:          P(첫단어) × P(둘째|첫째) × P(셋째|첫째,둘째) × ...
```

**왜 "Auto-regressive"인가?**

```
regressive = 회귀 (이전 값으로 다음 값 예측)
auto = 자기 자신

→ "자기 자신의 이전 출력으로 다음을 예측"

x₁ → x₂ → x₃ → x₄ → ...
     ↑     ↑     ↑
    x₁   x₁,x₂  x₁,x₂,x₃
```

**대표적인 Auto-regressive 모델:**

| 모델 | 생성 대상 | 순서 |
|------|----------|------|
| GPT | 텍스트 | 왼쪽 → 오른쪽 (단어 순서) |
| WaveNet | 오디오 | 과거 → 미래 (시간 순서) |
| PixelCNN | 이미지 | 좌상단 → 우하단 (픽셀 순서) |

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

### Whitening (백색화)

**Whitening이란?** 데이터를 "표준 정규분포"로 변환하는 것

**왜 "White"인가?**
```
White noise = 모든 주파수 성분이 균일한 노이즈
            = 어느 방향으로든 같은 분산

Whitening = 데이터를 이런 "균일한" 상태로 만들기
          = 타원형 분포 → 원형 분포
```

**수식:**

$$\mathbf{z} = \Sigma^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$$

```
변환 과정:

1. (x - μ): 중심을 원점으로 이동
2. Σ^(-1/2): 타원을 원으로 만듦 (늘어난 방향을 압축)

결과: z ~ N(0, I)
     = 모든 방향에서 분산 1, 상관관계 0
```

**왜 중요한가?**
```
• Diffusion: 노이즈 추가 후 최종 상태가 N(0,I)
• VAE: 잠재 공간을 N(0,I)로 정규화
• 전처리: 학습 안정화 (모든 feature가 비슷한 스케일)
```

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

**인코더:** $q_\phi(z|x)$
- "$x$가 주어졌을 때, $z$의 분포는?"
- 보통 $z|x \sim \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$

**디코더:** $p_\theta(x|z)$
- "$z$가 주어졌을 때, $x$의 분포는?"

**Prior:** $p(z) = \mathcal{N}(0, I)$
- "잠재 공간의 사전 분포"

**학습 목표:**

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

### Diffusion 모델이란?

**핵심 아이디어:** 데이터에 노이즈를 조금씩 추가하다가, 다시 제거하는 법을 학습

```
Forward (노이즈 추가) - 학습 안 함, 그냥 수학적으로 정의:
    이미지 → 약간 노이즈 → 더 노이즈 → ... → 완전 노이즈

Reverse (노이즈 제거) - 이걸 신경망으로 학습!:
    완전 노이즈 → 조금 복원 → 더 복원 → ... → 이미지
```

### Diffusion의 Forward Process

**수식:**

$$x_0 \sim p_{\text{data}}, \quad x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

| 항 | 의미 |
|---|---|
| $\sqrt{\bar{\alpha}_t}$ | 원본 비율 |
| $\sqrt{1-\bar{\alpha}_t}$ | 노이즈 비율 |

**$t$가 커지면:**
- $\sqrt{\bar{\alpha}_t} \to 0$ (원본 사라짐)
- $\sqrt{1-\bar{\alpha}_t} \to 1$ (노이즈만 남음)

**$t \to \infty$:**

$$x_T \sim \mathcal{N}(0, I) \quad \text{(거의 순수 노이즈)}$$

이게 왜 Gaussian? → CLT! 노이즈가 누적되면서 정규분포로 수렴

### Normalizing Flow 생성 모델

Real NVP, Glow 등의 구조:

$$z \sim \mathcal{N}(0, I) \xrightarrow{f_1} \xrightarrow{f_2} \cdots \xrightarrow{f_K} x$$

### Affine Coupling Layer란?

**문제:** 신경망은 보통 역변환이 불가능합니다.

$$y = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

$x \to y$는 쉽지만, $y \to x$는 방정식을 풀어야 해서 거의 불가능합니다.

**해결:** 입력을 반으로 나누면 **신경망을 쓰면서도 역변환 가능**

```
순방향:
    x₁' = x₁                      ← 절반은 그대로 통과
    x₂' = x₂ ⊙ s(x₁) + t(x₁)      ← 나머지 절반만 변환

역방향:
    x₁ = x₁'                      ← 그대로
    x₂ = (x₂' - t(x₁)) / s(x₁)    ← x₁을 알면 바로 계산!
```

**핵심 아이디어:**
- $s(\cdot), t(\cdot)$는 아무리 복잡한 신경망이어도 OK
- 절반을 "그대로 통과"시키니까, 그 값으로 나머지 절반을 역산 가능
- 여러 층을 쌓되, 매 층마다 어떤 절반을 고정할지 번갈아 바꿈

**장점:**
- 정확한 likelihood 계산 가능
- 학습과 샘플링 모두 효율적

**단점:**
- 표현력의 한계 (역변환 가능해야 함)
- 잠재 공간 차원 = 데이터 차원 (압축 불가)

### Reparameterization Trick

VAE 학습의 핵심 기법

**문제: 샘플링은 미분 불가능!**

$$z \sim \mathcal{N}(\mu, \sigma^2)$$

"확률적으로 뽑기"는 수학 연산이 아닙니다. 난수 생성기를 호출하는 것입니다.

```
미분 가능한 것:
    z = 2x + 3     →  ∂z/∂x = 2  ✓

미분 불가능한 것:
    z = random()   →  ∂z/∂??? = ???  ✗
```

**왜 문제인가?**

인코더가 $\mu=5, \sigma=2$를 출력했다고 하면:

$$z \sim \mathcal{N}(5, 4) \quad \Rightarrow \quad z = 6.3 \text{ (예시)}$$

질문: $\mu$를 $5.1$로 바꾸면 $z$가 얼마나 변하나?
답: **모른다!** 다시 뽑으면 완전히 다른 값이 나올 수도 있습니다.

```
loss → ... → z → μ, σ → 인코더
              ↑
          여기서 gradient가 끊김!
```

**해결: 샘플링을 "수식"으로 바꾸기**

$$\epsilon \sim \mathcal{N}(0, 1), \quad z = \mu + \sigma \cdot \epsilon$$

- $\epsilon$은 $\mu, \sigma$와 무관하게 미리 뽑음
- $z = \mu + \sigma \cdot \epsilon$은 그냥 곱셈, 덧셈 (미분 가능!)

**구체적 예시:**

$$\epsilon = 0.65 \text{ (미리 뽑음)}$$
$$z = 5 + 2 \times 0.65 = 6.3$$

$\mu$를 $5.1$로 바꾸면?

$$z = 5.1 + 2 \times 0.65 = 6.4 \quad \text{(정확히 0.1 증가!)}$$

이제 gradient가 흐릅니다:

$$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$

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
