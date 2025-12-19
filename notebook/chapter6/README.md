# Chapter 6: Entropy and Information Theory

> **책 페이지**: 163-196
> **핵심 주제**: 조건부 확률, 베이즈 규칙, 엔트로피, KL Divergence, 상호 정보량, Autoencoder
> **KAIST Challenge 연결**: Challenge 11 (KL Divergence), Challenge 12 (Information Bottleneck)

---

## 📚 목차

1. [왜 정보 이론인가?](#1-왜-정보-이론인가)
2. [조건부 확률과 베이즈 규칙](#2-조건부-확률과-베이즈-규칙)
3. [엔트로피: 불확실성의 측정](#3-엔트로피-불확실성의-측정)
4. [KL Divergence: 분포 간 거리](#4-kl-divergence-분포-간-거리)
5. [상호 정보량](#5-상호-정보량)
6. [Autoencoder와 정보 압축](#6-autoencoder와-정보-압축)
7. [Notebooks 가이드](#7-notebooks-가이드)
8. [Generative AI에서의 응용](#8-generative-ai에서의-응용)

---

## 1. 왜 정보 이론인가?

### 열역학에서 AI로

> **책 원문 (p.163):**
> "Entropy first appeared in thermodynamics as a measure of disorder... Shannon's fundamental insight was that the same mathematical structure captures the uncertainty of an information source."

```
열역학:
    엔트로피 = 시스템의 무질서도
    높은 엔트로피 = 무질서, 예측 불가

정보 이론:
    엔트로피 = 불확실성의 양
    높은 엔트로피 = 정보량이 많음, 예측 어려움
```

### Generative AI에서의 역할

| 개념 | AI에서의 역할 |
|------|-------------|
| **엔트로피** | 모델 예측의 불확실성 측정 |
| **KL Divergence** | VAE, Diffusion의 학습 목표 |
| **상호 정보량** | 표현 학습, Information Bottleneck |
| **Cross-Entropy** | 분류 손실함수 |

> **책 원문 (p.163):**
> "In diffusion models, entropy controls the trade-off between randomness and structure during denoising."

---

## 2. 조건부 확률과 베이즈 규칙

### 조건부 확률

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**직관**:
```
"B가 일어났다는 걸 알 때, A가 일어날 확률"

예: 비가 올 확률 = 30%
    구름이 꼈을 때 비가 올 확률 = 70%
```

### 베이즈 규칙

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| 용어 | 의미 | 예시 |
|------|------|------|
| $P(A)$ | 사전 확률 (Prior) | 병에 걸릴 확률 |
| $P(B|A)$ | 가능도 (Likelihood) | 병이 있을 때 양성일 확률 |
| $P(A|B)$ | 사후 확률 (Posterior) | 양성일 때 병이 있을 확률 |
| $P(B)$ | 증거 (Evidence) | 양성일 확률 |

### 의료 진단 예시 (책 Example 6.1.1)

```
설정:
    π = P(병) = 0.01 (1%)
    s = P(양성|병) = 0.95 (민감도)
    c = P(음성|건강) = 0.98 (특이도)

질문: 양성 판정을 받았을 때, 정말 병이 있을 확률은?

계산:
    P(양성) = s·π + (1-c)·(1-π)
            = 0.95×0.01 + 0.02×0.99
            = 0.0095 + 0.0198 = 0.0293

    P(병|양성) = (s·π) / P(양성)
               = 0.0095 / 0.0293
               ≈ 32.4%

놀랍죠? 95% 정확한 검사인데도, 양성이어도 실제 병일 확률은 32%!
이유: 병 자체가 희귀하기 때문 (Prior가 낮음)
```

### Naive Bayes vs Neural Network

> **책 원문 (p.168):**
> "Naïve Bayes models the likelihood as a product of conditionally independent features."

```
Naive Bayes:
    P(F|C) = ∏ᵢ P(Fᵢ|C)
    "픽셀들이 서로 독립" (실제론 아님!)

    장점: 계산 빠름, 해석 쉬움
    단점: 독립 가정이 비현실적

Neural Network:
    P(C|F) = softmax(NN(F))
    "복잡한 의존성 학습"

    장점: 높은 정확도
    단점: 해석 어려움
```

---

## 3. 엔트로피: 불확실성의 측정

### 정의

> **책 원문 (p.171):**
> "For a discrete random variable X with probability mass function P(X), the (Shannon) entropy is:"

$$H(X) = -\sum_{x} P(x) \log P(x)$$

연속 변수 (Differential Entropy):
$$H(X) = -\int p(x) \log p(x) \, dx$$

### 직관적 이해

```
동전 던지기:
    공정한 동전: P(H)=0.5, P(T)=0.5
    H = -0.5 log 0.5 - 0.5 log 0.5 = 1 bit
    "결과를 알려면 1비트 필요"

    편향된 동전: P(H)=0.99, P(T)=0.01
    H ≈ 0.08 bit
    "거의 확실하니 정보량 적음"

극단적 경우:
    확실한 결과 (P(A)=1): H = 0 (불확실성 없음)
    균등 분포 (n개 결과): H = log n (최대 불확실성)
```

### Predictive Entropy (예측 엔트로피)

분류 모델의 예측에 대한 엔트로피:

$$H_{pred}(x) = -\sum_{y} \hat{p}(y|x) \log \hat{p}(y|x)$$

```
낮은 예측 엔트로피:
    [0.99, 0.01, 0.00, ...] → 자신 있는 예측

높은 예측 엔트로피:
    [0.15, 0.14, 0.12, 0.11, ...] → 불확실한 예측
```

### MNIST에서의 예측 엔트로피

노트북 `Entropy-Classification-Experiment.ipynb`에서:

```
낮은 엔트로피 이미지:
    - 깔끔한 숫자, 모델이 확신
    - 예: 선명한 "1", "7"

높은 엔트로피 이미지:
    - 애매한 숫자, 여러 클래스와 비슷
    - 예: "4" vs "9", "3" vs "5"
```

---

## 4. KL Divergence: 분포 간 거리

### 정의

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

연속:
$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### 핵심 성질

```
1. 항상 0 이상: D_KL(P||Q) ≥ 0
2. P = Q일 때만 0
3. 비대칭: D_KL(P||Q) ≠ D_KL(Q||P)
4. 삼각 부등식 불만족 → 진짜 "거리"는 아님
```

### 직관적 의미

```
D_KL(P || Q) = "Q를 사용해서 P를 설명하는 비효율성"

예:
P = 실제 데이터 분포
Q = 모델이 예측하는 분포

D_KL(P || Q)가 크면:
    "모델이 실제를 잘 설명 못함"

D_KL(P || Q) = 0이면:
    "완벽한 모델!"
```

### 비대칭성의 의미

```
D_KL(P || Q):
    P가 있는 곳에 Q가 없으면 → 큰 페널티
    "mode covering" 경향

D_KL(Q || P):
    Q가 있는 곳에 P가 없으면 → 큰 페널티
    "mode seeking" 경향
```

### Cross-Entropy와의 관계

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

```
Cross-Entropy 최소화 = KL Divergence 최소화
(H(P)는 상수이므로)

이것이 왜 분류 문제에서 Cross-Entropy Loss를 쓰는 이유!
```

---

## 5. 상호 정보량

### 정의

$$I(X; Y) = H(X) - H(X|Y)$$

또는:
$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

### 직관

```
I(X; Y) = "Y를 알면 X에 대한 불확실성이 얼마나 줄어드나?"

예:
    X = 내일 날씨
    Y = 오늘 날씨

    오늘 날씨를 알면 내일 날씨 예측이 더 쉬워짐
    → I(X; Y) > 0

독립인 경우:
    I(X; Y) = 0
    "Y를 알아도 X에 대해 새로운 정보 없음"
```

### 벤 다이어그램 해석

```
        H(X)          H(Y)
     ┌───────┐    ┌───────┐
     │       │    │       │
     │   ┌───┼────┼───┐   │
     │   │   │    │   │   │
     │   │   │I(X;Y)   │   │
     │   │   │    │   │   │
     │   └───┼────┼───┘   │
     │       │    │       │
     └───────┘    └───────┘

I(X;Y) = H(X) ∩ H(Y) = 공유 정보
```

### Information Bottleneck

표현 학습의 핵심 원리:

$$\min_{Z} I(X; Z) - \beta \cdot I(Z; Y)$$

```
목표: 입력 X에서 표현 Z를 추출할 때

I(X; Z) 최소화:
    "Z가 X에 대한 정보를 너무 많이 갖지 않게"
    → 압축, 노이즈 제거

I(Z; Y) 최대화:
    "Z가 레이블 Y를 예측하는 데 유용하게"
    → 중요한 정보 보존

β로 균형 조절
```

---

## 6. Autoencoder와 정보 압축

### Autoencoder 구조

```
입력 x (784차원)
    ↓
  인코더 (압축)
    ↓
잠재 표현 z (32차원)
    ↓
  디코더 (복원)
    ↓
재구성 x̂ (784차원)

목표: x ≈ x̂
```

### PCA vs Autoencoder (Ch.1 연결)

| 방법 | 변환 | 장점 |
|------|------|------|
| **PCA (SVD)** | 선형: $z = V_k^\top x$ | 해석 가능, 최적 보장 |
| **Autoencoder** | 비선형: $z = f_\theta(x)$ | 더 강력한 압축 |

### 정보 이론적 해석

```
인코더: X → Z
    I(X; Z) ≤ H(X)
    "정보 손실 발생"

디코더: Z → X̂
    복원 오차 = 손실된 정보

좋은 Autoencoder:
    I(Z; X)를 최대화하면서
    Z의 차원은 최소화
```

### VAE의 KL 항

VAE의 ELBO:

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

```
첫째 항: 재구성 품질
    "z에서 x를 얼마나 잘 복원하나?"

둘째 항 (KL): 정규화
    "인코딩된 분포가 prior (N(0,I))와 얼마나 다른가?"

KL 항의 역할:
    - 잠재 공간을 구조화
    - 과적합 방지
    - 생성을 가능하게 함 (z ~ N(0,I)에서 샘플링)
```

---

## 7. Notebooks 가이드

### Info/ 폴더

| 노트북 | 뭘 배우나? | 핵심 실습 |
|--------|-----------|----------|
| `Bayes-Toy-Discrete.ipynb` | 베이즈 규칙 | 의료 진단 예시 |
| `Bayes-MNIST-NaiveBayes.ipynb` | Naive Bayes | MNIST 분류 |
| `Entropy-Classification-Experiment.ipynb` | 예측 엔트로피 | 불확실성 분석 |
| `CNN-MNIST-channel.ipynb` | CNN 채널 | 특징 시각화 |
| `autoencoder-entropy.ipynb` | Autoencoder | 압축과 정보 |
| `entropyNN.ipynb` | NN과 엔트로피 | 정보 흐름 |
| `UNet-MNIST-light.ipynb` | U-Net | Diffusion 준비 |
| `Wasserstein-1D-Gaussians.ipynb` | Wasserstein 거리 | 분포 비교 |
| `Hopfield-vs-Hamming.ipynb` | 연상 메모리 | 에너지 기반 |

### 꼭 해볼 실험들

**1. 예측 엔트로피 분석**
```python
# Entropy-Classification-Experiment.ipynb
# 높은/낮은 엔트로피 이미지 비교
# 왜 특정 이미지가 불확실한지 분석
```

**2. Naive Bayes vs NN**
```python
# Bayes-MNIST-NaiveBayes.ipynb
# Confusion matrix 비교
# 어떤 숫자 쌍을 헷갈리는가?
```

**3. Autoencoder 압축률**
```python
# autoencoder-entropy.ipynb
# 잠재 차원: 2, 8, 32, 128
# 재구성 품질 vs 압축률 트레이드오프
```

---

## 8. Generative AI에서의 응용

### VAE의 ELBO

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

```
ELBO 최대화 = 두 가지 동시 최적화:

1. 재구성: p(x|z)로 x를 잘 복원
2. 정규화: q(z|x)가 p(z)=N(0,I)에 가깝게

β-VAE:
    ELBO = E[log p(x|z)] - β·D_KL(q||p)

    β > 1: 더 강한 정규화 → 더 분리된 잠재 공간
    β < 1: 더 좋은 재구성 → 덜 구조화된 잠재 공간
```

### Diffusion의 ELBO

Diffusion도 ELBO로 해석 가능:

$$\log p(x_0) \geq \mathbb{E}\left[\sum_{t} -D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))\right]$$

```
각 step의 KL:
    Forward: q(x_{t-1}|x_t, x_0) (알려진 분포)
    Reverse: p_θ(x_{t-1}|x_t) (학습할 분포)

"모델이 각 denoising step을 얼마나 잘 배웠나"
```

### Cross-Entropy Loss

분류 문제의 표준 손실:

$$\mathcal{L} = -\sum_i y_i \log \hat{y}_i$$

```
y = [0, 0, 1, 0, ...] (one-hot 정답)
ŷ = [0.1, 0.05, 0.8, 0.05, ...] (모델 예측)

L = -log(0.8) ≈ 0.22

정답 클래스의 확률이 높을수록 Loss 낮음
```

### GAN의 정보 이론적 해석

```
Generator: z → G(z)
    목표: P_G ≈ P_data

Discriminator: D(x)
    목표: P_data와 P_G 구별

수렴 시:
    D_KL(P_data || P_G) = 0
    (이상적으로)
```

### Contrastive Learning

```
InfoNCE Loss:
    L = -log [exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ)]

정보 이론적 해석:
    I(view1; view2)의 하한을 최대화
    "같은 이미지의 다른 augmentation은 정보를 공유해야"
```

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **엔트로피 = 불확실성**
   - $H(X) = -\sum P(x) \log P(x)$
   - 높은 엔트로피 = 예측 어려움

2. **KL Divergence = 분포 차이**
   - $D_{KL}(P \| Q)$: Q로 P를 설명하는 비효율성
   - VAE, Diffusion의 학습 목표

3. **Cross-Entropy = 분류 손실**
   - $H(P, Q) = H(P) + D_{KL}(P \| Q)$
   - CE 최소화 ≈ KL 최소화

4. **상호 정보량 = 공유 정보**
   - $I(X; Y) = H(X) - H(X|Y)$
   - 표현 학습의 기준

5. **Information Bottleneck**
   - 압축하되 중요한 정보는 보존
   - Autoencoder의 원리

---

## 🔗 다른 챕터와의 연결

| 연결 | 설명 |
|------|------|
| **Ch.1 → Ch.6** | PCA = 선형 정보 압축 |
| **Ch.5 → Ch.6** | 확률 분포 → 엔트로피 계산 |
| **Ch.6 → Ch.7** | 엔트로피 → 확률 과정의 복잡도 |
| **Ch.6 → Ch.8** | KL → VAE, Energy-Based Models |
| **Ch.6 → Ch.9** | ELBO → Diffusion 학습 |

---

*이 문서는 Mathematics of Generative AI Book Chapter 6의 학습 가이드입니다.*
