# Chapter 9: Synthesis - Score-Based Diffusion and Beyond

> **책 페이지**: 266+
> **핵심 주제**: Score-Based Diffusion, Bridge Diffusion, VAE/GAN의 Diffusion 해석, Phase Transitions, RL, GFlowNets
> **KAIST Challenge 연결**: Challenge 19 (Score Matching), Challenge 20 (Diffusion Models)

---

## 📚 목차

1. [Score-Based Diffusion Models](#1-score-based-diffusion-models)
2. [Bridge Diffusion과 Schrödinger Bridge](#2-bridge-diffusion과-schrödinger-bridge)
3. [VAE와 GAN의 Diffusion 해석](#3-vae와-gan의-diffusion-해석)
4. [Dynamic Phase Transitions](#4-dynamic-phase-transitions)
5. [Stochastic Optimal Control과 RL](#5-stochastic-optimal-control과-rl)
6. [Generative Flow Networks (GFlowNets)](#6-generative-flow-networks-gflownets)
7. [Notebooks 가이드](#7-notebooks-가이드)
8. [전체 책 통합 요약](#8-전체-책-통합-요약)

---

## 1. Score-Based Diffusion Models

### 핵심 아이디어

> **책 원문 (p.266):**
> "Score-Based Diffusions currently represent the state of the art... a forward-time process that incrementally corrupts ground-truth samples by adding noise, and a reverse-time process that reconstructs data by gradually removing noise."

```
Forward Process (노이즈 추가):
    x₀ (데이터) → x₁ → x₂ → ... → x_T ≈ N(0, I)

Reverse Process (디노이징):
    x_T (노이즈) → ... → x₁ → x₀ (생성된 데이터)
```

### Forward SDE

$$dx_t = f(x_t, t) dt + g(t) dw_t$$

```
f(x, t): drift (보통 선형)
g(t): diffusion coefficient
w_t: Brownian motion

예: Variance Preserving (VP) SDE
    dx = -½ β(t) x dt + √β(t) dw
```

### Reverse SDE

$$dy_t = [f(y_t, t) - g(t)^2 \nabla \log p(y_t, t)] dt + g(t) d\bar{w}_t$$

**핵심**: Score function $\nabla \log p(x, t)$가 필요!

### Score Matching

```
Score: s(x, t) = ∇_x log p(x, t)

"확률이 증가하는 방향"

학습 목표:
    min_θ E[||s_θ(x_t, t) - ∇ log p(x_t|x_0)||²]

Denoising Score Matching:
    s(x_t, t) ≈ (x_0 - x_t) / σ_t²
    "노이즈가 추가된 방향의 역방향"
```

### Anderson's Theorem

> **책 원문 (p.268):**
> "By imposing that both dynamics describe the same time-marginal p(x, t), one sees that the two Fokker–Planck equations are consistent."

```
Forward와 Reverse의 marginal 분포가 일치!
    p_forward(x, t) = p_reverse(x, t)

이것이 Diffusion이 정확한 생성 모델인 이유
```

### 학습 파이프라인

```python
# Training
for x_0 in dataset:
    t = random_time()
    noise = random_gaussian()
    x_t = forward(x_0, t, noise)

    predicted_noise = score_net(x_t, t)
    loss = ||predicted_noise - noise||²

# Generation
x_T = random_gaussian()
for t in reversed(times):
    x_{t-1} = reverse_step(x_t, score_net, t)
return x_0
```

---

## 2. Bridge Diffusion과 Schrödinger Bridge

### Schrödinger Bridge 문제

> **책 원문 (p.272):**
> "Can we keep T finite while also making statistics of x(T) fixed?"

```
Standard Diffusion:
    x(0) ~ p_data
    x(T) ~ N(0, I)  (T → ∞에서)

Bridge Diffusion:
    x(0) ~ p_data
    x(1) = x_target (고정!)
    T = 1로 유한
```

### Doob's h-transform

$$dx(t) = [f(t; x) + G \nabla_x \log p(x(1)|x(t))] dt + \sqrt{G} dw_t$$

```
추가된 drift: G ∇ log p(x(1)|x(t))

"목표 x(1)을 향해 유도"
→ 모든 경로가 x(1)에서 끝남!
```

### Optimal Transport 연결

```
Schrödinger Bridge ↔ Entropic Optimal Transport

ε → 0:
    확률 경로 → 결정론적 OT map

GAN과의 연결:
    GAN = Zero-noise Schrödinger Bridge
    단일 스텝, 결정론적 변환
```

---

## 3. VAE와 GAN의 Diffusion 해석

### 통합된 관점

> **책 원문 (p.272):**
> "Many pre-diffusion generative models can now be reinterpreted as special or limiting cases within the broader diffusion framework."

```
Diffusion
    │
    ├── VAE = One-step diffusion (양방향)
    │       인코더: 노이즈 추가
    │       디코더: 디노이징
    │
    └── GAN = One-step reverse-only diffusion
            Generator: z → x
            단일 스텝 결정론적 변환
```

### VAE as Diffusion

```
VAE:
    Encoder: x → z ~ N(μ(x), σ(x)²)
    Decoder: z → x̂

Diffusion 해석:
    Forward: x에 노이즈 추가 → z
    Reverse: z에서 x 복원

차이:
    VAE: 단일 스텝, 학습된 노이즈
    Diffusion: 다단계, 고정된 스케줄
```

### GAN as Diffusion

> **책 원문 (p.273):**
> "GANs can be seen as a limiting case of Schrödinger bridges... as the noise level ε → 0, the bridge converges to a deterministic map."

```
GAN Generator:
    z ~ N(0, I) → x = G(z)

Diffusion 해석:
    T = 1
    g(t) = 0 (노이즈 없음)
    결정론적 reverse

"OT map을 adversarial하게 학습"
```

### 계층적 VAE → Discrete Diffusion

```
Ladder VAE:
    z_T → z_{T-1} → ... → z_1 → x

각 레이어:
    z_{t-1} = f(z_t) + noise

T → ∞:
    연속 시간 Diffusion으로 수렴!
```

---

## 4. Dynamic Phase Transitions

### U-Turn Diffusion

> **책 원문 (p.277):**
> "In U-Turn Diffusion, a pre-trained score-based diffusion model is modified by terminating the forward noising process at an intermediate time T_u."

```
U-Turn 아이디어:
    Forward: x_0 → x_{T_u}
    Reverse: x_{T_u} → y_0

T_u에 따른 행동 변화:
    작은 T_u: y_0 ≈ x_0 (거의 복원)
    큰 T_u: y_0 ≈ 새로운 샘플
```

### Phase Transitions

```
T_m (Memorization Time):
    이전: GT 샘플에 가까움
    이후: GT에서 벗어남

T_s (Speciation Time):
    이전: 같은 클래스
    이후: 다른 클래스로 점프!
```

### 물리학과의 연결

```
Spin Glass 이론:
    - Collapse transition (응축)
    - Separation transition (분리)

Diffusion에서:
    - T_m: 특정 데이터로 응축
    - T_s: 클래스 간 분리
```

---

## 5. Stochastic Optimal Control과 RL

### MDP (Markov Decision Process) 기초

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t) | s_0 = s\right]$$

```
s: 상태 (state)
a: 행동 (action)
r: 보상 (reward)
π: 정책 (policy)
V: 가치 함수 (value function)
```

### Bellman Equation

$$V^*(s) = \max_a \left[r(s, a) + \gamma \mathbb{E}_{s'}[V^*(s')]\right]$$

"최적 가치 = 즉각 보상 + 미래 가치의 기대값"

### Control as Inference

> **책 원문 (p.267):**
> "Physics-inspired priors (e.g. control as inference) enrich classical RL."

```
RL 목표:
    max E[Σ r_t]

Inference 관점:
    p(τ) ∝ exp(Σ r_t)

"좋은 trajectory는 높은 확률"
→ Sampling 문제로 변환!
```

### Diffusion + RL

```
세 가지 연결:
1. Diffusion의 reverse = Stochastic control
2. Score matching = Policy optimization
3. Denoising = Value function approximation
```

---

## 6. Generative Flow Networks (GFlowNets)

### GFlowNets 아이디어

> **책 원문 (p.267):**
> "Generative Flow Networks... samplers over decision trajectories rather than over raw data."

```
목표: p(x) ∝ R(x)에서 샘플링

Trajectory:
    s_0 → s_1 → ... → s_n = x

Flow Matching:
    각 상태로 들어오는 flow = 나가는 flow

"비가역적 생성 과정의 일반화"
```

### GFlowNets vs Diffusion

| 항목 | GFlowNets | Diffusion |
|------|-----------|-----------|
| **Time axis** | 구조적 (노드 추가) | 연속적 (노이즈) |
| **Reversibility** | 비가역적 | 가역적 |
| **Structure** | DAG (discrete) | 연속 공간 |
| **응용** | 조합 최적화 | 이미지 생성 |

### Decision Flow (GFlowNets의 확장)

```
"Diffusion + GFlowNets"

연속 시간 + 구조적 생성

응용:
    - 분자 설계
    - 강화학습
    - 조합 최적화
```

---

## 7. Notebooks 가이드

### 주요 노트북 (다른 폴더에 분산)

| 노트북 | 위치 | 내용 |
|--------|------|------|
| `02-SGM-with-SDE-9grid.ipynb` | chapter9/ | Score-based diffusion |
| `ring_vae_latent_diffusion_comparison.ipynb` | chapter9/ | VAE vs Diffusion |
| `Langevin-DoubleWell.ipynb` | chapter7/ | Langevin dynamics |
| `RBM-MCMC.ipynb` | chapter7/ | Energy-based sampling |

### 핵심 실습

**1. Score-Based Diffusion**
```python
# 02-SGM-with-SDE-9grid.ipynb
# 9개 Gaussian mode에서 diffusion
# Forward/Reverse SDE 시각화
```

**2. U-Turn 실험**
```python
# T_u 변화에 따른 생성 품질
# Memorization vs Generation 경계
```

**3. VAE-Diffusion 비교**
```python
# ring_vae_latent_diffusion_comparison.ipynb
# 같은 데이터셋에서 두 방법 비교
```

---

## 8. 전체 책 통합 요약

### 수학적 기초 (Ch.1-3)

```
Ch.1 선형대수:
    SVD, 고유값 분해, 텐서
    → 데이터 표현, 차원 축소

Ch.2 미적분/ODE:
    미분, Jacobian, 동적 시스템
    → Neural ODE, SDE

Ch.3 최적화:
    Gradient Descent, 정규화, SGD
    → 신경망 학습
```

### 신경망 아키텍처 (Ch.4)

```
MLP → CNN → ResNet → Neural ODE → Transformer

핵심 통찰:
    ResNet = ODE의 이산화
    Skip connection = Gradient flow 보장
```

### 확률론적 기초 (Ch.5-6)

```
Ch.5 확률:
    분포, 변환, CLT
    → 생성 모델의 수학적 기반

Ch.6 정보 이론:
    엔트로피, KL, 상호정보량
    → VAE의 ELBO, Cross-entropy loss
```

### 확률 과정과 샘플링 (Ch.7)

```
샘플링: ITS, Importance Sampling
과정: Brownian Motion, SDE, Markov Chain
방법: MCMC, Langevin Dynamics

→ Diffusion의 직접적 기반!
```

### 구조적 모델 (Ch.8)

```
Energy-Based Models:
    p(x) ∝ exp(-E(x))

Graphical Models:
    조건부 독립 구조

VAE:
    Variational Inference + Neural Network
```

### 최종 통합 (Ch.9)

```
Score-Based Diffusion:
    Ch.7 SDE + Ch.4 Neural Net + Ch.6 KL

모든 생성 모델의 통합:
    VAE = One-step diffusion
    GAN = Deterministic bridge
    Flow = ODE-based diffusion

"모든 길은 Diffusion으로 통한다"
```

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **Score Function이 핵심**
   - $s(x,t) = \nabla_x \log p(x,t)$
   - 확률이 증가하는 방향

2. **Forward-Reverse 대응**
   - Forward: 노이즈 추가 (정의됨)
   - Reverse: Score로 디노이징 (학습)

3. **모든 생성 모델의 통합**
   - VAE, GAN = Diffusion의 특수 경우
   - Bridge = 유한 시간 diffusion

4. **Phase Transitions**
   - T_m: Memorization
   - T_s: Speciation
   - 물리학과 AI의 연결

5. **Control + Generation**
   - RL ↔ Diffusion
   - GFlowNets: 구조적 생성

---

## 🔗 전체 책 연결도

```
Ch.1 Linear Algebra ─────────────────┐
         │                           │
Ch.2 Calculus/ODE ────────┐         │
         │                │         │
Ch.3 Optimization ────────┼─→ Ch.4 Neural Networks
                          │              │
Ch.5 Probability ─────────┼─→ Ch.7 Stochastic Processes
         │                │              │
Ch.6 Information Theory ──┼─→ Ch.8 Energy-Based Models
                          │              │
                          └──────→ Ch.9 Synthesis
                                   (Score-Based Diffusion)
```

---

*이 문서는 Mathematics of Generative AI Book Chapter 9의 학습 가이드이자 전체 책의 통합 요약입니다.*
