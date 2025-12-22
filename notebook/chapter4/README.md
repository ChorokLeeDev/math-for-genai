# Chapter 4: Neural Networks & Deep Learning

> **책 페이지**: 99-132
> **핵심 주제**: 신경망 메커니즘, CNN, ResNet, Neural ODE, 표현 학습
> **KAIST Challenge 연결**: Challenge 7 (ResNet & Skip Connections), Challenge 8 (Neural ODE)

---

## 🎯 The Evolution: Perceptron → CNN → ResNet → Neural ODE

이 챕터의 핵심은 **신경망 아키텍처의 진화**를 이해하는 것입니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    신경망 아키텍처의 진화 과정                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ① Perceptron (1958)                                                        │
│     y = sign(w⊤x + b)                                                       │
│     • 단일 뉴런, 선형 분리만 가능                                              │
│     • 한계: XOR 문제 해결 불가                                                │
│              ↓                                                              │
│              ↓  "비선형 활성화 + 다층 구조"                                    │
│              ↓                                                              │
│  ② CNN (1989, LeNet)                                                        │
│     h = σ(Conv(x) + b)                                                      │
│     • 국소 연결 + 가중치 공유                                                 │
│     • 이미지의 공간적 구조 활용                                               │
│     • 한계: 층이 깊어지면 gradient vanishing                                  │
│              ↓                                                              │
│              ↓  "Skip Connection으로 gradient 경로 확보"                      │
│              ↓                                                              │
│  ③ ResNet (2015)                                                            │
│     h_{t+1} = h_t + F(h_t)                                                  │
│     • Identity mapping + 잔차 학습                                           │
│     • 100층 이상도 안정적 학습                                               │
│     • 통찰: "이건 ODE의 Euler method 아닌가?"                                 │
│              ↓                                                              │
│              ↓  "이산 → 연속의 극한"                                          │
│              ↓                                                              │
│  ④ Neural ODE (2018)                                                        │
│     dh/dt = f(h, t; θ)                                                      │
│     • 연속적인 깊이, 무한 층의 극한                                           │
│     • 메모리 O(1), Adjoint method                                           │
│     • Diffusion, Normalizing Flow의 수학적 기반                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 각 단계의 핵심 돌파구

#### ① Perceptron → MLP: "XOR 문제"

**문제:** 단일 퍼셉트론은 선형 분리만 가능

```
XOR 문제:

(0,1)●         ●(1,1)     ← 출력 1
      \       /
       \     /
        \   /               하나의 직선으로 분리 불가!
         \ /
(0,0)○─────────○(1,0)     ← 출력 0
```

**해결:** 은닉층 + 비선형 활성화

```
입력 → [은닉층] → 출력

은닉층이 "공간을 비틀어서" 선형 분리 가능하게 만듦

예: XOR을 풀 수 있는 2층 네트워크
    h₁ = σ(x₁ + x₂ - 0.5)    ← "OR" 역할
    h₂ = σ(x₁ + x₂ - 1.5)    ← "AND" 역할
    y  = σ(h₁ - h₂ - 0.5)    ← h₁=1, h₂=0일 때만 1
```

**수학적 핵심:** Universal Approximation Theorem
- 충분히 넓은 은닉층 하나면 어떤 연속 함수든 근사 가능

---

#### ② MLP → CNN: "파라미터 폭발"

**문제:** 이미지에 Fully Connected를 쓰면?

```
224×224×3 컬러 이미지 → 첫 번째 FC 층 (1000 뉴런)

파라미터 수 = 224 × 224 × 3 × 1000 = 1.5억 개!

문제:
• 메모리 부족
• 과적합 위험
• 위치 정보 무시 (고양이가 어디 있든 같은 처리)
```

**해결:** 국소 연결 + 가중치 공유

```
Convolution:

    3×3 필터 하나로 전체 이미지 스캔

    ┌───────────────┐
    │ 이미지        │
    │    ┌───┐     │    같은 필터가 슬라이딩
    │    │필터│───→ │    → 파라미터: 단 9개!
    │    └───┘     │
    └───────────────┘

장점:
• 파라미터 대폭 감소 (1.5억 → 수천 개)
• 평행이동 불변: 고양이가 어디 있든 "고양이"로 인식
```

**수학적 핵심:** Convolution = 평행이동 불변 연산

$$(f * g)(x) = \int f(t) \cdot g(x-t) \, dt$$

---

#### ③ CNN → ResNet: "Gradient 소멸"

**문제:** 층을 깊게 쌓으면 학습이 안 됨

```
Chain Rule로 gradient 계산:

∂L/∂W₁ = ∂L/∂h_L × ∂h_L/∂h_{L-1} × ... × ∂h_2/∂h_1 × ∂h_1/∂W₁

각 ∂h_{k+1}/∂h_k < 1이면?
    0.9^50 ≈ 0.005  → Gradient Vanishing!

각 ∂h_{k+1}/∂h_k > 1이면?
    1.1^50 ≈ 117    → Gradient Exploding!
```

**해결:** Skip Connection

```
기존:    h_{t+1} = F(h_t)

ResNet:  h_{t+1} = h_t + F(h_t)
                   ↑
               Skip Connection

Gradient:
    ∂h_{t+1}/∂h_t = I + ∂F/∂h_t
                    ↑
                항상 1 이상 보장!
```

**핵심 통찰:** Identity $I$가 "gradient 고속도로" 역할

```
50층을 통과해도:
    gradient ≥ 1 × 1 × ... × 1 = 1  (최소 보장)

→ 100층, 1000층도 학습 가능!
```

---

#### ④ ResNet → Neural ODE: "이산에서 연속으로"

**관찰:** ResNet을 자세히 보면...

$$h_{t+1} = h_t + F(h_t)$$

이건 ODE의 **Euler method**와 같은 구조!

$$x(t + \Delta t) = x(t) + f(x(t)) \cdot \Delta t$$

**극한:** 층 수 $L \to \infty$, 각 변화량 $\to 0$

```
ResNet (이산):   h_0 → h_1 → h_2 → ... → h_L

                        ↓ L → ∞

Neural ODE (연속): dh/dt = f(h, t; θ)
                   h(0) ────────────→ h(T)
```

**장점:**

| 항목 | ResNet | Neural ODE |
|------|--------|------------|
| 메모리 | 모든 $h_t$ 저장 $(O(L))$ | $h(0), h(T)$만 $(O(1))$ |
| 깊이 | 고정 (50층, 101층...) | 연속적 (시간 $T$ 조절) |
| 해석 | 이산적 변환 | 연속적 흐름 (물리학적 해석)

**수학적 핵심:**

$$\frac{dh}{dt} = f(h, t; \theta)$$

---

### 요약 표

| 단계 | 문제 | 해결책 | 수학적 핵심 |
|------|------|--------|-------------|
| **Perceptron → MLP** | XOR 불가 (선형 한계) | 은닉층 + 비선형 | Universal Approximation |
| **MLP → CNN** | 파라미터 1.5억 개 | 3×3 필터 공유 | Convolution = 평행이동 불변 |
| **CNN → ResNet** | $0.9^{50} \approx 0$ | Skip: $h + F(h)$ | $\frac{\partial}{\partial h} = I + ...$ |
| **ResNet → Neural ODE** | 이산적 층 | 연속 미분방정식 | $\frac{dh}{dt} = f(h,t)$ |

### 핵심 해결 도구 간단 소개

**① Universal Approximation Theorem**
> "충분히 넓은 은닉층 하나면 어떤 연속 함수든 근사 가능"

```
f(x) ≈ Σᵢ cᵢ · σ(wᵢ⊤x + bᵢ)

• σ = 비선형 활성화 (ReLU, sigmoid 등)
• 이론적 보장이지만, "얼마나 넓어야?" → 실용적으론 깊이가 더 효율적
```

**② Convolution (합성곱)**
> "필터를 이미지 위로 슬라이딩하며 특징 추출"

```
(f * g)[i,j] = Σₕ,ᵥ f[h,w] · g[i-h, j-w]

• 평행이동 불변성: 고양이가 어디에 있든 "고양이"로 인식
• 가중치 공유: 같은 필터를 전체 이미지에 재사용 → 파라미터 절약
```

**③ Skip Connection (잔차 연결)**
> "입력을 출력에 직접 더해서 gradient 흐름 보장"

```
h_{t+1} = h_t + F(h_t)    (ResNet)
          ↑
      Identity shortcut

• Gradient: ∂h_{t+1}/∂h_t = I + ∂F/∂h_t
• I (항등행렬)가 있어서 gradient가 최소 1은 보장됨
• 100층 이상에서도 학습 가능하게 만든 핵심
```

**④ Adjoint Method**
> "ODE의 역전파를 메모리 효율적으로 계산"

```
Forward:  h(0) → ODE Solver → h(T)
Backward: a(T) → Adjoint ODE → a(0), ∂L/∂θ

• 중간 상태 저장 불필요 → 메모리 O(1)
• 일반 backprop은 O(L) 메모리 필요
• Neural ODE 학습의 핵심 기술
```

### 왜 이 순서로 배워야 하는가?

```
Perceptron을 이해해야 → "왜 층을 쌓는가?" 이해
CNN을 이해해야      → "왜 구조가 중요한가?" 이해
ResNet을 이해해야   → "왜 ODE와 연결되는가?" 이해
Neural ODE를 이해해야 → Diffusion Model의 수학적 기반 이해
```

---

## 📚 목차

1. [신경망이 뭔가요?](#1-신경망이-뭔가요)
2. [Neural Network Mechanics](#2-neural-network-mechanics)
3. [CNN: 이미지를 위한 신경망](#3-cnn-이미지를-위한-신경망)
4. [ResNet: Skip Connection의 힘](#4-resnet-skip-connection의-힘)
5. [Neural ODE: 연속적인 깊이](#5-neural-ode-연속적인-깊이)
6. [Notebooks 가이드](#6-notebooks-가이드)
7. [Generative AI에서의 응용](#7-generative-ai에서의-응용)

---

## 1. 신경망이 뭔가요?

### 앞 챕터들과의 연결

> **책 원문 (p.99):**
> "We begin this chapter by situating neural networks within the mathematical framework developed in Chapters 1,2,3."

신경망은 앞서 배운 모든 것의 **집합체**입니다:

| 챕터 | 개념 | 신경망에서의 역할 |
|------|------|------------------|
| **Ch.1** | 행렬, 텐서, 선형 변환 | 각 층의 가중치, 입력 데이터 표현 |
| **Ch.2** | ODE, 동적 시스템 | ResNet = ODE의 이산화 |
| **Ch.3** | 최적화, GD | 손실함수 최소화로 학습 |

### 신경망의 본질

```
신경망 = "층을 쌓아서 복잡한 함수를 만드는 것"

입력 x → [층1] → [층2] → ... → [층L] → 출력 ŷ

각 층:  h_{k+1} = σ(W_k · h_k + b_k)

여기서:
- W_k: 가중치 행렬 (학습됨)
- b_k: 편향 벡터 (학습됨)
- σ: 활성화 함수 (ReLU, tanh 등)
```

### 왜 "Deep" Learning인가?

```
얕은 신경망:  입력 → [층1] → 출력
             간단한 패턴만 학습 가능

깊은 신경망:  입력 → [층1] → [층2] → ... → [층100] → 출력
             계층적으로 복잡한 패턴 학습

             층1: 선 검출
             층2: 모서리, 질감
             층3: 부분 (눈, 코)
             층L: 전체 개념 (얼굴)
```

---

## 2. Neural Network Mechanics

### Perceptron: 시작점 (1958)

> **책 원문 (p.101):**
> "A perceptron takes an input vector x ∈ ℝ^d, computes a linear score w⊤x + b, and then applies a threshold nonlinearity."

```
ŷ = sign(w⊤x + b)

문제: 선형 분리 가능한 문제만 풀 수 있음!
      XOR 문제조차 못 품
```

**Minsky & Papert (1969)**: "단층 퍼셉트론은 XOR을 못 푼다!"
→ 이게 **다층 신경망**의 필요성을 부각시킴

### 단일 은닉층 신경망

```
입력 (x₁, x₂)
    ↓
은닉층: h = tanh(W₁x + b₁)   ← 10개 뉴런
    ↓
출력층: ŷ = σ(W₂h + b₂)      ← 확률 출력
```

**핵심 차이: 특징 학습 (Feature Learning)**

| 방법 | 특징 | 장단점 |
|------|------|--------|
| **로지스틱 회귀 + 다항식** | 손으로 설계: $\phi(x) = [x_1, x_2, x_1^2, x_1x_2, ...]$ | 도메인 지식 필요 |
| **신경망** | 자동 학습: $h = \sigma(Wx + b)$ | 데이터에서 특징 발견 |

> **책 원문 (p.104):**
> "A neural network introduces a learned nonlinear feature map through its hidden layers... Feature engineering is replaced by representation learning."

### Interpolation vs Extrapolation

**흥미로운 관찰** (책 Example 4.1.2):

```
데이터: 5개 점, 훈련 구간 [-1, 1]

다항식 회귀 (차수 9):
- Interpolation: 훈련 점은 완벽히 통과
- 하지만: 진동이 심함 (Runge 현상)
- Extrapolation: 완전히 발산

신경망 (50개 은닉 뉴런):
- Interpolation: 부드러운 곡선
- 파라미터 >> 데이터인데도 과적합 없음!
- Extrapolation: 역시 실패 (외삽은 어려움)
```

**핵심 교훈**:
- 신경망은 **interpolation**에서 놀랍도록 안정적
- 하지만 **extrapolation**은 어떤 모델도 어렵다
- Over-parameterization이 반드시 과적합을 의미하지 않음

---

## 3. CNN: 이미지를 위한 신경망

### 왜 Fully Connected가 이미지에 부적합한가?

```
28×28 MNIST 이미지를 FC로 처리하면?
입력: 784차원
첫 층 (100 뉴런): 784 × 100 = 78,400 파라미터

224×224 컬러 이미지라면?
입력: 224 × 224 × 3 = 150,528차원
첫 층: 1.5억 파라미터?! → 불가능
```

### CNN의 핵심 아이디어

> **책 원문 (p.107):**
> "Convolutional Neural Networks exploit exactly this kind of prior structure: locality and weight sharing."

| 원리 | 설명 | 효과 |
|------|------|------|
| **국소성 (Locality)** | 각 뉴런은 작은 영역만 봄 | 파라미터 대폭 감소 |
| **가중치 공유 (Weight Sharing)** | 같은 필터를 전체 이미지에 적용 | 위치 불변성 |
| **계층적 특징** | 얕은 층: 엣지 → 깊은 층: 객체 | 추상화 학습 |

### CNN 구조 예시 (MNIST)

```
입력: 28×28×1
    ↓
Conv1: 16 필터, 3×3, padding=1 → 28×28×16
MaxPool: 2×2 → 14×14×16
    ↓
Conv2: 32 필터, 3×3, padding=1 → 14×14×32
MaxPool: 2×2 → 7×7×32
    ↓
Flatten: 1568차원
FC1: 128 뉴런
FC2: 10 뉴런 (클래스)
    ↓
Softmax → 확률 분포
```

### 컨볼루션 연산 복습 (Ch.1 연결)

$$y_{i,j} = \sum_{h,w} k_{h,w} \cdot x_{i+h, j+w}$$

```
입력 이미지:          커널 (필터):
┌───┬───┬───┬───┐    ┌───┬───┐
│ 1 │ 2 │ 3 │ 0 │    │ 1 │ 0 │
├───┼───┼───┼───┤    ├───┼───┤
│ 4 │ 5 │ 6 │ 1 │    │ 0 │-1 │
├───┼───┼───┼───┤    └───┴───┘
│ 7 │ 8 │ 9 │ 0 │
├───┼───┼───┼───┤
│ 1 │ 0 │ 2 │ 3 │
└───┴───┴───┴───┘

출력 (0,0) = 1×1 + 2×0 + 4×0 + 5×(-1) = -4
```

### 채널 수 선택 원칙

> **책 원문 (p.110):**

1. **얕은 층**: 간단한 특징 (엣지, 코너) → 16-64 채널
2. **깊은 층**: 복잡한 특징 (객체 부분) → 128-256+ 채널
3. **경험적 규칙**: 해상도가 반으로 줄면, 채널은 두 배로

---

## 4. ResNet: Skip Connection의 힘

### 문제: 깊은 신경망의 학습 어려움

```
층을 깊게 쌓으면 좋을까?

20층 신경망:  정확도 92%
56층 신경망:  정확도 89%  ← 오히려 나빠짐!

왜? Gradient Vanishing/Exploding
깊은 층의 gradient가 앞쪽까지 전달이 안 됨
```

### ResNet의 해결책: Residual Block

> **책 원문 (p.113):**
> "The key idea is to make each layer learn a residual correction on top of an identity map."

```
기존 방식:
    h_{t+1} = F(h_t)
    "h_t를 완전히 새로운 h_{t+1}로 변환"

ResNet 방식:
    h_{t+1} = h_t + F(h_t)
    "h_t에 작은 변화량 F(h_t)만 더함"
         ↑
    Skip Connection (Identity Mapping)
```

**왜 이게 효과적인가?**

```
Jacobian 관점:

기존: ∂h_{t+1}/∂h_t = ∂F/∂h_t
      → 행렬 곱이 누적되면 폭발/소멸

ResNet: ∂h_{t+1}/∂h_t = I + ∂F/∂h_t
        → Identity I가 gradient 전달을 보장!
```

### ResNet = ODE의 이산화!

**핵심 통찰** (책 p.114):

$$h_{t+1} = h_t + F(h_t, \theta_t)$$

이건 ODE의 **Euler method**:

$$x(t+\Delta t) = x(t) + f(x(t)) \cdot \Delta t$$

| ResNet | ODE |
|--------|-----|
| 층 인덱스 $t$ | 시간 $t$ |
| $h_t$ (층 출력) | $x(t)$ (상태) |
| $F(h_t)$ | $f(x(t)) \cdot \Delta t$ |
| 층 통과 | 시간 1 스텝 |

---

## 5. Neural ODE: 연속적인 깊이

### ResNet에서 Neural ODE로

> **책 원문 (p.114):**
> "If we imagine L large and the changes per layer small, this composition suggests a continuous-depth limit."

```
ResNet (이산적):
    h_0 → h_1 → h_2 → ... → h_L
    "L개의 층을 차례로 통과"

Neural ODE (연속적):
    dh(t)/dt = f(h(t), t; θ)
    h(0) → ∫ → h(T)
    "연속적인 변환"
```

### Neural ODE의 장점

| 항목 | ResNet | Neural ODE |
|------|--------|------------|
| **메모리** | 모든 중간 $h_t$ 저장 | $h(0)$, $h(T)$만 저장 |
| **깊이** | 고정 (예: 50층) | 적분 시간 $T$ 조절 |
| **Gradient** | Backprop through layers | Adjoint method |
| **해석** | 이산적 변환 | 연속적 흐름 |

### Adjoint Method: 메모리 효율적 학습

문제: $h(T)$에서 $\theta$로 어떻게 gradient를?

**일반적 Backprop**: 모든 중간 상태 저장 → 메모리 $O(L)$

**Adjoint Method**:
1. Forward: $h(0) \to h(T)$ 계산
2. Backward: 별도의 ODE로 gradient 계산

$$\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f}{\partial h}$$

메모리: $O(1)$ (상수!)

### 실습: Spiral 데이터셋

노트북 `NeuralODE-Spiral.ipynb`에서:
- 2D spiral 데이터 분류
- Neural ODE가 연속적으로 데이터를 분리하는 과정 시각화
- ResNet과 비교

---

## 6. Notebooks 가이드

### NN/ 폴더

| 노트북 | 뭘 배우나? | 핵심 실습 |
|--------|-----------|----------|
| `LogReg+NN-supervised-2D.ipynb` | NN vs 로지스틱 회귀 | 2D 분류 비교 |
| `InterPoly-vs-NN.ipynb` | 보간 vs 외삽 | 다항식 vs NN 비교 |
| `cnn-simple-MNIST.ipynb` | CNN 기초 | MNIST 분류 |
| `CNN-MNIST-PCA.ipynb` | CNN + 차원축소 | 특징 시각화 |
| `ResNet9-Spiral.ipynb` | ResNet 구조 | Skip connection 효과 |
| `NeuralODE-Spiral.ipynb` | Neural ODE | 연속적 변환 |
| `NeuralODE-Adjoint-Spiral.ipynb` | Adjoint method | 메모리 효율적 학습 |
| `NN-Decoding.ipynb` | 신경망 디코딩 | 정보 이론 연결 |

### 꼭 해볼 실험들

**1. 은닉층 크기 실험**
```python
# 은닉 뉴런 수: 5, 10, 50, 100
# 어떻게 decision boundary가 바뀌나?
```

**2. Skip Connection 효과**
```python
# ResNet9-Spiral.ipynb에서
# skip connection 제거하면?
# 학습이 얼마나 어려워지나?
```

**3. Neural ODE vs ResNet**
```python
# 같은 문제를 두 방법으로 풀어보기
# 메모리 사용량, 정확도 비교
```

---

## 7. Generative AI에서의 응용

### Diffusion Model의 U-Net

```
Diffusion의 핵심 신경망 = U-Net

인코더 (다운샘플링):
    64×64 → 32×32 → 16×16 → 8×8

디코더 (업샘플링):
    8×8 → 16×16 → 32×32 → 64×64

Skip Connection으로 인코더-디코더 연결!
→ 고해상도 디테일 보존
```

### ResNet이 Diffusion에서 중요한 이유

```
Diffusion의 각 denoising step:
    x_t → [U-Net] → x_{t-1}

U-Net 내부에 ResNet Block 사용:
- 깊은 네트워크 안정적 학습
- Gradient flow 보장
- 노이즈 제거 품질 향상
```

### Neural ODE → Continuous Normalizing Flow

```
Normalizing Flow = 확률 분포 변환

이산적: z → f_1 → f_2 → ... → f_K → x

연속적 (CNF):
    dz/dt = f(z, t)

장점:
- 임의의 변환 가능 (역변환 가능할 필요 없음)
- 메모리 효율적
```

### Transformer의 Skip Connection

```
Transformer Block도 ResNet 구조!

Attention:
    x = x + Attention(LayerNorm(x))

FFN:
    x = x + FFN(LayerNorm(x))

수백 층도 안정적 학습 가능
→ GPT, BERT, LLM의 기반
```

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **신경망 = Ch.1,2,3의 종합**
   - 행렬 변환 + 동적 시스템 + 최적화

2. **특징 학습 (Representation Learning)**
   - 손으로 설계 → 데이터에서 자동 학습

3. **CNN = 이미지의 구조 활용**
   - 국소성 + 가중치 공유 + 계층적 특징

4. **ResNet = Skip Connection**
   - $h_{t+1} = h_t + F(h_t)$
   - 깊은 네트워크의 학습 가능하게 함

5. **Neural ODE = 연속적 신경망**
   - ResNet의 연속 버전
   - 메모리 효율적, 해석 가능

---

## 🔗 다른 챕터와의 연결

| 연결 | 설명 |
|------|------|
| **Ch.1 → Ch.4** | 행렬 연산이 각 층의 기본 |
| **Ch.2 → Ch.4** | ResNet = ODE의 Euler method |
| **Ch.3 → Ch.4** | SGD/Adam으로 가중치 학습 |
| **Ch.4 → Ch.6** | Autoencoder, U-Net (정보 압축) |
| **Ch.4 → Ch.7** | Neural ODE → SDE → Diffusion |
| **Ch.4 → Ch.9** | Diffusion의 backbone |

---

*이 문서는 Mathematics of Generative AI Book Chapter 4의 학습 가이드입니다.*
