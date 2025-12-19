# Chapter 8: Energy Based (Graphical) Models

> **책 페이지**: 235-264
> **핵심 주제**: 에너지 함수, 그래픽 모델, Bayesian Networks, Variational Inference, VAE, RBM, GNN
> **KAIST Challenge 연결**: Challenge 16 (Variational Inference), Challenge 17 (VAE), Challenge 18 (Graph Neural Networks)

---

## 📚 목차

1. [에너지 기반 모델이란?](#1-에너지-기반-모델이란)
2. [Graphical Models의 분류](#2-graphical-models의-분류)
3. [Bayesian Networks](#3-bayesian-networks)
4. [Variational Inference와 ELBO](#4-variational-inference와-elbo)
5. [Mean-Field Approximation](#5-mean-field-approximation)
6. [Variational Auto-Encoder (VAE)](#6-variational-auto-encoder-vae)
7. [Graph Neural Networks (GNN)](#7-graph-neural-networks-gnn)
8. [Generative AI에서의 응용](#8-generative-ai에서의-응용)

---

## 1. 에너지 기반 모델이란?

### 에너지와 확률의 연결

> **책 원문 (p.235):**
> "In AI, energy is used metaphorically to define a scalar function over configurations... where lower energy corresponds to higher probability."

$$p(x) = \frac{1}{Z} \exp(-E(x))$$

```
E(x): 에너지 함수
    낮은 에너지 = 높은 확률 (선호되는 상태)
    높은 에너지 = 낮은 확률 (비선호 상태)

Z: 분할 함수 (Partition Function)
    Z = Σₓ exp(-E(x))
    모든 상태의 합 → 계산이 어려움!
```

### 물리학에서 AI로

| 물리학 | AI/ML |
|--------|-------|
| 에너지 | 비용 함수, 음의 로그 확률 |
| 낮은 에너지 상태 | 데이터 manifold 위의 점 |
| 온도 | 샘플링의 다양성 조절 |
| Boltzmann 분포 | 모델 분포 |

### 왜 에너지 기반인가?

```
장점:
1. 정규화 상수 없이 에너지 정의 가능
   E(x)만 모델링하면 됨 (Z 계산 불필요)

2. 그래프로 분해 가능
   E(x) = Σᵢ Eᵢ(xᵢ) + Σᵢⱼ Eᵢⱼ(xᵢ, xⱼ)

단점:
샘플링이 어려움 (MCMC 필요)
```

---

## 2. Graphical Models의 분류

### 그래프로 표현하는 확률 모델

> **책 원문 (p.237):**
> "Several classes of graphical models have become central in AI research."

```
노드: 확률 변수
엣지: 변수 간 의존성

그래프 구조 → 분포 분해 규칙 결정
```

### 주요 Graphical Models

| 모델 | 그래프 타입 | 특징 |
|------|-----------|------|
| **Bayesian Networks** | 방향 비순환 (DAG) | 인과 관계 모델링 |
| **Markov Random Fields** | 무방향 | 지역 의존성 |
| **Factor Graphs** | 이분 그래프 | 명시적 분해 |
| **Hidden Markov Model** | 체인 구조 | 시계열 |

### Ising Model: 가장 간단한 예

```
변수: x = (x₁, ..., xₙ), xᵢ ∈ {-1, +1}

에너지: E(x) = -Σᵢⱼ Jᵢⱼ xᵢ xⱼ - Σᵢ hᵢ xᵢ

Jᵢⱼ: 이웃 스핀 간 상호작용
hᵢ: 외부 자기장

높은 온도: 스핀 무질서
낮은 온도: 정렬된 상태 (상전이)
```

---

## 3. Bayesian Networks

### 구조

> **책 원문 (p.238):**
> "In a Bayesian network, the joint distribution factorizes according to the network structure."

$$P(X_1, ..., X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))$$

```
예: A → B, A → C

P(A, B, C) = P(A) × P(B|A) × P(C|A)

A가 B와 C의 공통 원인
```

### 예시: 의료 진단 (책 Example 8.1.1)

```
A: 질병 유무 (0 or 1)
B: 증상 1 (A에 의존)
C: 증상 2 (A에 의존)

P(A=1) = 0.3
P(B=1|A=1) = 0.8,  P(B=1|A=0) = 0.2
P(C=1|A=1) = 0.7,  P(C=1|A=0) = 0.4

P(A=1, B=1, C=1) = 0.3 × 0.8 × 0.7 = 0.168
```

### Hidden Markov Model (HMM)

```
숨겨진 상태: S₁ → S₂ → S₃ → ...
         ↓     ↓     ↓
관측:      O₁    O₂    O₃

P(S, O) = P(S₁) P(O₁|S₁) × Πₜ P(Sₜ|Sₜ₋₁) P(Oₜ|Sₜ)

응용: 음성 인식, 시계열 분석
```

---

## 4. Variational Inference와 ELBO

### 문제: 사후 분포 계산

$$p(z|x) = \frac{p(x|z) p(z)}{p(x)} = \frac{p(x|z) p(z)}{\int p(x|z) p(z) dz}$$

"분모의 적분이 intractable!"

### 해결: 변분 근사

> **책 원문 (p.241):**
> "Variational inference posits a surrogate distribution q(x|θ) and seeks to find parameters θ that minimize the KL divergence."

$$\min_\theta D_{KL}(q(x|\theta) \| p(x))$$

### ELBO (Evidence Lower Bound)

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

첫 항: 재구성 품질
둘째 항: 정규화

ELBO 최대화 ≈ KL(q||p) 최소화
```

### ELBO 증명 스케치

```
Jensen's Inequality:
    log E[X] ≥ E[log X]

적용:
    log p(x) = log Σ_z p(x,z)
             = log Σ_z q(z) × p(x,z)/q(z)
             = log E_q[p(x,z)/q(z)]
             ≥ E_q[log(p(x,z)/q(z))]
             = E_q[log p(x|z)] + E_q[log p(z)/q(z)]
             = E_q[log p(x|z)] - KL(q||p)
```

---

## 5. Mean-Field Approximation

### 아이디어

> **책 원문 (p.242):**
> "Under mean-field, the surrogate distribution takes a fully factorized form."

$$q(x) = \prod_i q_i(x_i)$$

"각 변수가 독립이라고 가정" (실제로는 상관 있지만!)

### Ising Model에서의 Mean-Field

```
실제: p(x) ∝ exp(Σᵢⱼ Jᵢⱼ xᵢxⱼ + Σᵢ hᵢxᵢ)
      스핀들이 서로 영향

Mean-Field: q(x) = Πᵢ q(xᵢ)
            각 스핀 독립

자기 일관성 방정식:
    mᵢ = tanh(hᵢ + Σⱼ Jᵢⱼ mⱼ)

mᵢ = E_q[xᵢ] = "평균 자화"
```

### Mean-Field의 한계

```
장점:
- 계산 효율적
- 닫힌 형태 업데이트

단점:
- 상관관계 무시
- 분산 과소추정
- 다봉 분포 잘 못 잡음
```

### Belief Propagation (BP)

```
Tree-structured graph에서:
    BP가 정확한 추론!

Loop가 있으면:
    Loopy BP → 근사

Bethe Approximation:
    q(x) = Π_edge q(xᵢ, xⱼ) / Π_node q(xᵢ)^(degree-1)
```

---

## 6. Variational Auto-Encoder (VAE)

### 구조

> **책 원문 (p.247):**
> "VAEs merges the ideas of variational inference from Bayesian statistics with deep neural network architectures."

```
인코더 (Encoder): q_φ(z|x)
    x → Neural Net → (μ, σ)
    z ~ N(μ, σ²)

디코더 (Decoder): p_θ(x|z)
    z → Neural Net → x̂

Prior: p(z) = N(0, I)
```

### VAE의 ELBO

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

```
첫 항 (Reconstruction):
    z에서 x를 잘 복원하나?
    Binary: BCE loss
    Continuous: MSE loss

둘째 항 (KL Regularization):
    인코더 출력이 N(0,I)에 가까운가?

    KL(N(μ,σ²) || N(0,1))
    = (1/2) Σ (μ² + σ² - log σ² - 1)
```

### Reparameterization Trick

```
문제: z ~ q(z|x)에서 θ로 gradient 못 흘림

해결:
    ε ~ N(0, 1)
    z = μ + σ × ε

    z가 이제 μ, σ의 함수!
    → backprop 가능
```

### 생성 과정

```
학습 후:
    1. z ~ N(0, I) 샘플링
    2. x = Decoder(z)
    → 새로운 데이터 생성!
```

---

## 7. Graph Neural Networks (GNN)

### 왜 그래프에 Neural Network?

```
일반 NN: 고정된 크기 입력 (이미지, 벡터)
GNN: 임의의 그래프 구조 입력

응용:
- 분자 구조 예측
- 소셜 네트워크
- 추천 시스템
- 물리 시뮬레이션
```

### Message Passing Framework

$$h_i^{(l+1)} = \text{UPDATE}\left(h_i^{(l)}, \text{AGGREGATE}(\{h_j^{(l)} : j \in \mathcal{N}(i)\})\right)$$

```
1. AGGREGATE: 이웃 노드의 정보 모음
   예: 평균, 합, max

2. UPDATE: 자신의 표현 업데이트
   예: MLP, GRU

k번 반복 → k-hop 이웃 정보 통합
```

### GCN (Graph Convolutional Network)

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

```
Ã = A + I (self-loop 추가)
D̃: degree matrix

"이웃의 가중 평균 + 선형 변환 + 활성화"
```

### Spectral vs Spatial

| 방식 | 아이디어 | 장단점 |
|------|---------|--------|
| **Spectral** | 그래프 푸리에 변환 | 이론적, 비효율적 |
| **Spatial** | Message passing | 효율적, 직관적 |

---

## 8. Generative AI에서의 응용

### VAE: 잠재 공간 생성

```
VAE의 잠재 공간:
    - 연속적, 구조화됨
    - 보간 가능 (z1과 z2 사이)
    - 조작 가능 (특정 방향 = 특정 속성)

한계:
    - 흐릿한 이미지 경향
    - KL collapse 문제
```

### VAE vs GAN vs Diffusion

| 모델 | 학습 목표 | 샘플링 | 품질 |
|------|----------|--------|------|
| **VAE** | ELBO 최대화 | z → Decoder | 흐릿함 |
| **GAN** | Adversarial | z → Generator | 선명, 불안정 |
| **Diffusion** | Score matching | 반복 denoising | SOTA |

### Energy-Based Models (EBMs)

```
E_θ(x): 학습된 에너지 함수

p(x) ∝ exp(-E_θ(x))

학습: Contrastive Divergence
샘플링: Langevin / MCMC

장점: 유연한 모델링
단점: 느린 샘플링
```

### GNN for Molecular Generation

```
분자 = 그래프
    노드: 원자
    엣지: 결합

GNN 생성:
    1. 노드 임베딩 학습
    2. Auto-regressive로 노드/엣지 추가
    또는
    VAE의 latent space → 분자

응용: 신약 설계, 재료 발견
```

### Score-Based Models (Ch.9 미리보기)

```
Score: s(x) = ∇_x log p(x)

에너지와의 관계:
    s(x) = -∇_x E(x)

Diffusion:
    Forward: 점점 noise 추가
    Reverse: Score로 denoise

"Energy-Based + Stochastic Process"
```

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **에너지 ↔ 확률**
   - $p(x) \propto \exp(-E(x))$
   - 낮은 에너지 = 높은 확률

2. **Graphical Models**
   - 그래프 = 조건부 독립 구조
   - 분포의 효율적 분해

3. **Variational Inference**
   - 어려운 분포를 쉬운 분포로 근사
   - ELBO 최대화

4. **VAE = Variational + Neural Network**
   - Encoder: q(z|x)
   - Decoder: p(x|z)
   - Reparameterization trick

5. **GNN = 그래프 위의 Neural Network**
   - Message passing
   - 구조적 데이터 처리

---

## 🔗 다른 챕터와의 연결

| 연결 | 설명 |
|------|------|
| **Ch.5 → Ch.8** | 확률 분포 → 에너지 해석 |
| **Ch.6 → Ch.8** | KL → ELBO, VAE |
| **Ch.7 → Ch.8** | MCMC → EBM 샘플링 |
| **Ch.4 → Ch.8** | Neural Net → VAE, GNN |
| **Ch.8 → Ch.9** | EBM/VAE → Diffusion 통합 |

---

*이 문서는 Mathematics of Generative AI Book Chapter 8의 학습 가이드입니다.*
