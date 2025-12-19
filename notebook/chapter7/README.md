# Chapter 7: Stochastic Processes

> **책 페이지**: 198-233
> **핵심 주제**: 샘플링, Importance Sampling, Brownian Motion, Markov Chain, MCMC, Auto-regressive Models
> **KAIST Challenge 연결**: Challenge 13 (Importance Sampling), Challenge 14 (Langevin Dynamics), Challenge 15 (MCMC)

---

## 📚 목차

1. [왜 확률 과정인가?](#1-왜-확률-과정인가)
2. [Exact Sampling](#2-exact-sampling)
3. [Importance Sampling](#3-importance-sampling)
4. [Brownian Motion과 Diffusion](#4-brownian-motion과-diffusion)
5. [Markov Chain과 MCMC](#5-markov-chain과-mcmc)
6. [Langevin Dynamics](#6-langevin-dynamics)
7. [Auto-regressive Models](#7-auto-regressive-models)
8. [Notebooks 가이드](#8-notebooks-가이드)
9. [Generative AI에서의 응용](#9-generative-ai에서의-응용)

---

## 1. 왜 확률 과정인가?

### 확률 분포에서 확률 과정으로

> **책 원문 (p.198):**
> "Stochastic processes provide the natural mathematical language for describing uncertainty... They are the backbone of sampling, inference, noise injection, model training, and generative mechanisms."

```
Chapter 5: 확률 분포
    "한 시점의 랜덤 변수"
    X ~ P(X)

Chapter 7: 확률 과정
    "시간에 따라 진화하는 랜덤 변수"
    X(t), t ∈ [0, T]
```

### Generative AI에서의 핵심 역할

| 확률 과정 | 응용 |
|----------|------|
| **Markov Chain** | Token-by-token 생성, MCMC |
| **Brownian Motion** | Diffusion 모델의 Forward Process |
| **Langevin Dynamics** | Score-based 생성 |
| **Auto-regressive** | GPT, Transformer |

---

## 2. Exact Sampling

### Inverse Transform Sampling (ITS)

> **책 원문 (p.199):**
> "Given U ~ Uniform(0,1), F⁻¹(U) is an exact sample from that distribution."

```
아이디어:
    균등분포 U ~ Uniform(0,1)
    CDF의 역함수 적용: X = F⁻¹(U)
    → X는 원하는 분포를 따름!

예: 지수분포 샘플링
    F(x) = 1 - e^(-λx)
    F⁻¹(u) = -log(1-u)/λ

    U ~ Uniform(0,1)
    X = -log(1-U)/λ ~ Exponential(λ)
```

### 왜 1D에서만 가능한가?

```
1D: CDF F: ℝ → [0,1]은 역변환 가능

다차원: CDF F: ℝ^d → [0,1]
    스칼라 하나에서 d차원 벡터 복원 불가!
```

### Chain Rule Sampling (다차원 확장)

```
p(x₁, x₂, ..., xₙ) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ...

순차적 샘플링:
    1. x₁ ~ p(x₁)           ← ITS 사용
    2. x₂ ~ p(x₂|x₁)        ← ITS 사용
    3. x₃ ~ p(x₃|x₁,x₂)     ← ITS 사용
    ...

이것이 Auto-regressive 모델의 원리!
```

---

## 3. Importance Sampling

### 문제: 복잡한 분포에서의 기댓값 계산

$$\mathbb{E}_{p}[f(x)] = \int f(x) p(x) dx$$

"p(x)에서 샘플링이 어렵다면?"

### 해결책: 다른 분포 q(x) 사용

> **책 원문 (p.203):**
> "Importance Sampling provides a principled way to compute expectations with respect to a target distribution p(x) by drawing samples from a simpler proposal distribution q(x)."

$$\mathbb{E}_p[f(x)] = \mathbb{E}_q\left[f(x) \cdot \frac{p(x)}{q(x)}\right]$$

```
IS 추정량:
    E_p[f(x)] ≈ (1/N) Σᵢ w(xᵢ) · f(xᵢ)

    w(xᵢ) = p(xᵢ) / q(xᵢ)  (중요도 가중치)

조건: q(x) > 0 wherever p(x) > 0
```

### 예시: 희귀 사건 확률

노트북 `ImportanceSampling-RareEvent.ipynb`:

```
목표: P(X > 3) 추정, X ~ N(0,1)

직접 Monte Carlo:
    10⁵개 샘플 중 ~135개만 x > 3
    분산 매우 큼!

Importance Sampling:
    q(x) = N(3, 1)  ← 3 근처에서 많이 샘플링
    w(x) = p(x)/q(x)

    분산 대폭 감소!
```

### Effective Sample Size (ESS)

```
ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²

ESS가 높으면: 효과적인 proposal
ESS가 낮으면: 가중치가 몇 개에 집중 → 추정 불안정
```

### Adaptive Importance Sampling

> **책 원문 (p.207):**
> "Adaptive Importance Sampling aims to iteratively adapt the proposal distribution to better approximate the target."

Cross-Entropy Method:
1. 초기 proposal q₀ 설정
2. 샘플링, 가중치 계산
3. 가중 MLE로 q 업데이트
4. 반복

---

## 4. Brownian Motion과 Diffusion

### Brownian Motion 정의

> **책 원문 (p.210):**
> "Brownian motion is the simplest nontrivial continuous-time stochastic process and serves as the universal scaling limit of random walks."

$$dW_t = \sqrt{dt} \cdot Z, \quad Z \sim \mathcal{N}(0, 1)$$

**핵심 성질**:
```
1. 연속 경로: W(t)는 연속 함수
2. 독립 증분: W(t+s) - W(s)는 W(t)와 독립
3. 가우시안 증분: W(t) - W(s) ~ N(0, t-s)
4. W(0) = 0
```

### Heat Equation과의 연결

```
Brownian Motion:
    입자가 랜덤하게 확산

Heat Equation:
    ∂u/∂t = (1/2) ∂²u/∂x²

연결:
    u(x, t) = E[f(W_t) | W_0 = x]
    "초기 조건 f의 확산 = 열 전파"
```

### SDE (Stochastic Differential Equation)

일반적인 SDE:

$$dX_t = f(X_t, t) dt + g(X_t, t) dW_t$$

```
f(X, t): drift (결정론적 방향)
g(X, t): diffusion (확률적 변동)
dW_t: Brownian motion 증분
```

### Fokker-Planck Equation

SDE의 확률 밀도 p(x, t) 진화:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}[f \cdot p] + \frac{1}{2}\frac{\partial^2}{\partial x^2}[g^2 \cdot p]$$

"입자들의 분포가 시간에 따라 어떻게 변하는가?"

---

## 5. Markov Chain과 MCMC

### Markov Chain

> **책 원문 (p.220):**
> "A Markov chain is a sequence of random variables X₀, X₁, X₂, ... where the distribution of X_{n+1} depends only on X_n."

$$P(X_{n+1} | X_n, X_{n-1}, ..., X_0) = P(X_{n+1} | X_n)$$

"미래는 과거 전체가 아닌 현재에만 의존"

### 정상 분포 (Stationary Distribution)

$$\pi = \pi P$$

"충분히 오래 돌리면, 어디서 시작하든 분포가 π로 수렴"

### MCMC의 아이디어

```
목표: π(x)에서 샘플링

문제: π(x)에서 직접 샘플링 어려움

해결: π를 정상 분포로 갖는 Markov Chain 설계
      → 충분히 오래 돌리면 π에서 샘플!
```

### Metropolis-Hastings Algorithm

```python
def MH(x, T):
    for t in range(T):
        # 1. Proposal
        x_new = proposal(x)

        # 2. Acceptance probability
        α = min(1, π(x_new) * q(x|x_new) / (π(x) * q(x_new|x)))

        # 3. Accept/Reject
        if random() < α:
            x = x_new

    return x
```

### Detailed Balance

$$\pi(x) P(x \to y) = \pi(y) P(y \to x)$$

"x에서 y로 가는 확률 흐름 = y에서 x로 오는 확률 흐름"

→ 이걸 만족하면 π가 정상 분포!

---

## 6. Langevin Dynamics

### Overdamped Langevin Equation

> **책 원문 (p.224):**

$$dX_t = -\nabla U(X_t) dt + \sqrt{2} dW_t$$

```
-∇U(X): 에너지가 낮은 쪽으로 이동 (gradient descent)
√2 dW: 랜덤한 탐색 (noise)

결과: X_t → π(x) ∝ exp(-U(x)) 로 수렴
```

### 에너지 기반 해석

```
에너지: U(x)
분포: π(x) ∝ exp(-U(x)/T)

낮은 에너지 = 높은 확률

온도 T:
    T → 0: 최소값에만 집중
    T → ∞: 균등 탐색
```

### Score Function과의 연결

$$\nabla \log p(x) = -\nabla U(x)$$

```
Score = 확률이 증가하는 방향
      = 에너지가 감소하는 방향

Langevin Dynamics:
    dX = ∇ log p(X) dt + √2 dW
    "Score 방향으로 이동 + noise"
```

### Double Well 예시

노트북 `Langevin-DoubleWell.ipynb`:

```
U(x) = (x² - 1)² / 4

두 개의 well: x = ±1

Langevin으로 샘플링하면:
- 두 well 사이를 왔다 갔다
- 각 well에서 머무는 시간 ∝ exp(-barrier height)
```

---

## 7. Auto-regressive Models

### Chain Rule 기반 생성

$$p(x_1, ..., x_n) = \prod_{i=1}^n p(x_i | x_1, ..., x_{i-1})$$

> **책 원문 (p.227):**
> "This is precisely the mechanism underlying auto-regressive generative models."

```
텍스트 생성:
    p("The" "cat" "sat") = p("The") × p("cat"|"The") × p("sat"|"The cat")

이미지 생성:
    p(픽셀1, 픽셀2, ...) = p(픽셀1) × p(픽셀2|픽셀1) × ...
```

### Markov Chain과의 차이

```
Markov Chain (차수 1):
    p(x_n | x_{n-1}, ..., x_1) = p(x_n | x_{n-1})
    "바로 이전만 봄"

Auto-regressive:
    p(x_n | x_{n-1}, ..., x_1)
    "모든 이전을 봄" → 더 표현력 높음

Transformer:
    Attention으로 모든 이전 토큰 참조 가능!
```

---

## 8. Notebooks 가이드

### Stochastic/ 폴더

| 노트북 | 뭘 배우나? | 핵심 실습 |
|--------|-----------|----------|
| `ITS-1D.ipynb` | Inverse Transform Sampling | 1D 샘플링 |
| `ChainRuleSampling-2D.ipynb` | Chain Rule 샘플링 | 2D 분포 생성 |
| `ImportanceSampling-RareEvent.ipynb` | IS 기초 | 희귀 사건 추정 |
| `ImportanceSampling-GaussianPosterior.ipynb` | Bayesian IS | 사후 분포 추정 |
| `AdaptiveIS-CE-fitGaussian.ipynb` | Adaptive IS | CE method |
| `AdaptiveIS-CE-rare-event.ipynb` | 희귀 사건 + AIS | 고급 샘플링 |
| `BrownianMotion-and-HeatEquation.ipynb` | Brownian Motion | 열 방정식 연결 |
| `Langevin-DoubleWell.ipynb` | Langevin Dynamics | 에너지 기반 샘플링 |
| `RBM-MCMC.ipynb` | MCMC | Restricted Boltzmann Machine |

### 꼭 해볼 실험들

**1. Importance Sampling 분산**
```python
# ImportanceSampling-RareEvent.ipynb
# 다른 proposal q(x)로 실험
# ESS 비교, 분산 비교
```

**2. Langevin 온도 효과**
```python
# Langevin-DoubleWell.ipynb
# 온도 T: 0.1, 0.5, 1.0, 2.0
# 모드 간 전이 빈도 관찰
```

**3. Chain Rule 순서 효과**
```python
# ChainRuleSampling-2D.ipynb
# (x₁, x₂) vs (x₂, x₁) 순서
# 결과 분포 동일한지 확인
```

---

## 9. Generative AI에서의 응용

### Diffusion Model = SDE

```
Forward SDE (노이즈 추가):
    dX_t = f(X_t, t) dt + g(t) dW_t

Reverse SDE (디노이징):
    dY_t = [f(Y_t, t) - g(t)² ∇ log p(Y_t, t)] dt + g(t) dW̄_t

핵심: Score ∇ log p(x, t)를 신경망으로 학습!
```

### Score-Based Generative Models

```
학습:
    Score network sθ(x, t) ≈ ∇ log p(x, t)

생성:
    Langevin dynamics로 샘플링:
    x_{n+1} = x_n + ε · sθ(x_n, t) + √(2ε) · z
```

### Auto-regressive + Transformer = GPT

```
GPT의 생성:
    1. <start> 토큰으로 시작
    2. p(token_1 | <start>) 에서 샘플링
    3. p(token_2 | <start>, token_1) 에서 샘플링
    ...

Masked Self-Attention:
    각 토큰은 이전 토큰들만 참조 가능
    → Chain Rule 구현!
```

### Markov Chain과 Token Generation

```
Top-k Sampling:
    확률 높은 k개 중에서 선택
    → "결정론적" ↔ "다양성" 트레이드오프

Temperature Scaling:
    p(token) ∝ exp(logit / T)

    T < 1: 더 결정론적 (높은 확률에 집중)
    T > 1: 더 다양함 (분포 평평해짐)
```

### MCMC in Energy-Based Models

```
Energy-Based Model:
    p(x) ∝ exp(-E_θ(x))

샘플링:
    MCMC (Langevin, HMC 등) 필요
    → 비용이 큼, Diffusion보다 느림

장점:
    모델이 explicit energy function
    이상치 탐지 등에 유용
```

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **Importance Sampling**
   - $w(x) = p(x)/q(x)$
   - 좋은 proposal 선택이 핵심

2. **Brownian Motion**
   - 연속 시간 랜덤 워크
   - Diffusion의 기초

3. **Langevin Dynamics**
   - $dX = -\nabla U(X) dt + \sqrt{2} dW$
   - Gradient + Noise → Target 분포 샘플링

4. **MCMC**
   - Detailed Balance → Stationary Distribution
   - 복잡한 분포에서 샘플링

5. **Auto-regressive**
   - Chain Rule 분해
   - GPT, Transformer의 원리

---

## 🔗 다른 챕터와의 연결

| 연결 | 설명 |
|------|------|
| **Ch.2 → Ch.7** | ODE → SDE (노이즈 추가) |
| **Ch.5 → Ch.7** | 확률 분포 → 시간에 따른 진화 |
| **Ch.6 → Ch.7** | 엔트로피 → 확률 과정의 복잡도 |
| **Ch.7 → Ch.8** | MCMC → Energy-Based Models |
| **Ch.7 → Ch.9** | Langevin/SDE → Diffusion Models |

---

*이 문서는 Mathematics of Generative AI Book Chapter 7의 학습 가이드입니다.*
