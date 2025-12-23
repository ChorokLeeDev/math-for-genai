# KAIST AI Mini-Course Challenge Solutions (Lectures 1-4)

> **목표**: Lecture 1-4의 12개 Challenge에 대한 상세 해설
>
> **참조 교재**: Mathematics of Generative AI (MathGenAIBook12_14_25.pdf)
>
> **기존 Lecture 1-2 해설**: [notebook/chapter1/KAIST-challenges-solutions.md](../chapter1/KAIST-challenges-solutions.md)

---

# Challenge 매핑 테이블

| Lecture | Challenge | 책 Exercise | 핵심 개념 |
|---------|-----------|-------------|-----------|
| **1** | 1 | Exercise 1.1.3 (Einstein Summation) | 텐서 축약, GPU 효율성 |
| **1** | 2 | Exercise 1.2.2 (SVD for Matrix Completion) | Low-Rank Approximation |
| **1** | 3 | Exercise 2.1.1 (AD and Jacobian) | Chain Rule, Backprop |
| **1** | 4 | Exercise 2.2.3 (ODE Regression) | 파라미터 추정 |
| **2** | 5 | Exercise 3.2.1 (GD Trajectories) | Convex/Non-Convex Landscapes |
| **2** | 6 | Exercise 3.5.1 (Tiny Transformers) | SGD, Adam, RMSProp |
| **2** | 7 | Exercise 4.1.3 (CNN Filter/Size) | CNN Capacity |
| **2** | 8 | Exercise 4.3.2 (PCA on Activations) | Representation Learning |
| **3-4** | 9 | Exercise 5.2.3 (1D Normalizing Flow) | Transport Maps |
| **3-4** | 10 | Exercise 5.3.4 (Empirical Multivariate) | Covariance Limitations |
| **3-4** | 11 | Exercise 6.1.2 (Diagnosing Model Failures) | Naive Bayes vs NN |
| **3-4** | 12 | Exercise 6.2.6 (Wasserstein Geometry) | W1 vs KL Divergence |

---

# Lecture 1 Challenges (Ch. 1-2: Linear Algebra & Calculus)

> **상세 해설**: [notebook/chapter1/KAIST-challenges-solutions.md](../chapter1/KAIST-challenges-solutions.md) 참조

## Challenge 1-4 요약

Challenges 1-4는 기존 해설 파일에서 자세히 다루고 있습니다. 핵심 요점만 정리합니다.

### Challenge 1: Einstein Notation
- **y = Ax**: $y_i = A_{ij}x_j$
- **Frobenius norm**: $\|A\|_F^2 = A_{ij}A_{ij}$
- **C_ikl = A_ij B_jkl**: C는 3차원 텐서 (I × K × L)
- **메모리 연속성**: GPU에서 coalesced access가 FLOPs보다 중요

### Challenge 2: SVD/PCA
- **90% 에너지 k**: $\sum_{i=1}^{k} \sigma_i^2 / \sum_{i=1}^{r} \sigma_i^2 \geq 0.90$
- **Trade-off**: k↑ = 정보↑, 메모리↑
- **공분산 SVD**: N >> D일 때 X^T X (D×D) 분해가 훨씬 효율적

### Challenge 3: Backpropagation
- **Jacobian**: $J = W_2 \cdot \text{diag}(\sigma'(z_1)) \cdot W_1$
- **∂L/∂W₁**: $\delta_1 \cdot x^T$ where $\delta_1 = (W_2^T \frac{\partial L}{\partial f}) \odot \sigma'(z_1)$
- **Forward/Backward 분리**: 인과관계와 중간값 저장 필요
- **Gradient Clipping**: 방향 유지하며 크기만 제한

### Challenge 4: ODE Parameter Estimation
- **ODE 기반 회귀**: ẋ를 직접 피팅하면 노이즈 증폭 방지
- **물리적 구조**: ODE가 데이터 변동보다 안정적인 관계 제공

---

# Lecture 2 Challenges (Ch. 3-4: Optimization & Deep Learning)

---

## Challenge 5: Optimization Landscapes as Vector Fields

> **책 참조**: Exercise 3.2.1 (Gradient Descent Trajectories in Convex and Non-Convex Landscapes)

### 배경 지식: 최적화의 기하학적 관점

#### Loss Landscape란?

손실 함수 L(θ)를 파라미터 공간의 "지형"으로 시각화:

```
    Loss (높이)
         │
         │    ⛰️ Local max
         │   ╱  ╲
         │  ╱    ╲
         │ ╱   ○  ╲     ← Saddle point
         │╱    ↓   ╲
    ────────────────────→ θ
              ⬇️
           Global min
```

#### Gradient Descent = 벡터장 따라가기

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

- **-∇L**: 손실이 가장 빠르게 감소하는 방향
- **η**: 학습률 (보폭)
- **궤적**: 벡터장을 따라 흐르는 "물줄기"

---

### Question 1: Convex Case - 왜 모든 궤적이 같은 점으로 수렴하는가?

#### Convex 함수의 정의

함수 f가 convex ⟺ 임의의 두 점을 잇는 선분이 함수 그래프 위에 있음

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

#### 핵심 성질: 단일 전역 최소점

**Convex 함수의 특징**:
1. 모든 local minimum = global minimum
2. 기울기가 0인 점이 유일 (있다면)
3. 어디서 시작해도 "내리막"을 따라가면 같은 곳에 도착

```
Convex (그릇 모양):        Non-convex (울퉁불퉁):
        │                         │
        │   ╲     ╱               │  ╲ ╱╲ ╱╲
        │    ╲   ╱                │   ╲╱  ╲╱
        │     ╲ ╱                 │    ↑   ↑
        │      ○                  │  local  local
        │   global                │   min   min
```

#### 학습률이 경로 기하학에 미치는 영향

**작은 학습률** (η 작음):
```
출발점 →  .  .  .  .  .  .  . → 최소점
         작은 스텝, 많은 반복
         부드러운 곡선 경로
```

**큰 학습률** (η 큼):
```
출발점 →    .       .       . → 최소점
            ↘     ↙
             큰 스텝, 적은 반복
             지그재그 가능
```

**너무 큰 학습률**:
```
출발점 →  .     .        .
           ↘   ↗   ↘
             발산!
```

#### Convex Quadratic의 특별한 경우

$$L(\theta) = \frac{1}{2}\theta^T H \theta - b^T \theta$$

**Gradient descent 업데이트**:
$$\theta_{t+1} = \theta_t - \eta(H\theta_t - b) = (I - \eta H)\theta_t + \eta b$$

**수렴 조건**: 모든 H의 고유값 λ에 대해 |1 - ηλ| < 1
$$\Rightarrow 0 < \eta < \frac{2}{\lambda_{max}}$$

**최적 학습률**: $\eta^* = \frac{2}{\lambda_{max} + \lambda_{min}}$

---

### Question 2: Non-convex Case - 다른 초기화, 다른 결과

#### Multiple Basins of Attraction

```
Non-convex landscape:
    Loss
      │
      │╲    ╱╲    ╱
      │ ╲  ╱  ╲  ╱
      │  ╲╱    ╲╱
      │  A      B      ← 두 개의 local minima
      └──────────────→ θ

시작점에 따라:
- 왼쪽에서 시작 → Basin A로 수렴
- 오른쪽에서 시작 → Basin B로 수렴
```

**초기화가 중요한 이유**:
- 같은 알고리즘도 시작점에 따라 다른 해에 도달
- Neural network의 랜덤 초기화 → 매번 다른 결과

#### SGD 노이즈의 역할

**Deterministic GD**:
```
정확히 -∇L 방향으로 이동
→ 가장 가까운 local minimum에 갇힘
```

**Stochastic GD (SGD)**:
```
-∇L + noise 방향으로 이동
→ 노이즈가 shallow minima를 탈출하게 도움
```

#### 노이즈가 돕는 세 가지 상황

**1. Saddle Point 탈출**

```
        ↑ Loss
        │    ════════
        │   ↗  ↙ (기울기 ≈ 0)
        │  ↗    ↙
        │ 노이즈가
        │ 옆으로 밀어줌
```

- Saddle point: 일부 방향으로 내려가고, 다른 방향으로 올라감
- 기울기 ≈ 0이므로 GD가 느림
- SGD 노이즈가 "옆으로 밀어서" 탈출 가속

**2. Shallow Minima 탈출**

```
Loss
  │     ╲    ╱╲       ╱
  │      ╲  ╱  ╲     ╱
  │       ╲╱    ╲   ╱
  │      shallow  ╲ ╱
  │                deep
  │
노이즈 진폭 > shallow barrier → 탈출 가능
```

- Shallow minimum: 장벽이 낮은 local minimum
- 적절한 노이즈가 장벽을 넘게 해줌
- Deep minimum에는 덜 영향 (장벽이 높으므로)

**3. Flat Minima 선호**

```
Sharp minimum:          Flat minimum:
  │╲    ╱│               │    ════    │
  │ ╲  ╱ │               │   ╱    ╲   │
  │  ╲╱  │               │  ╱      ╲  │
  │      │               │ ╱        ╲ │

노이즈에 의해           노이즈에도
쉽게 튕겨나감            안정적
```

- SGD의 암묵적 정규화: flat minima를 선호
- Flat minima → 일반화 성능이 좋은 경향

#### Python 코드: Convex vs Non-convex

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(grad_fn, x0, lr=0.1, n_steps=100, noise_scale=0):
    """Gradient descent with optional noise"""
    trajectory = [x0.copy()]
    x = x0.copy()

    for _ in range(n_steps):
        g = grad_fn(x)
        noise = noise_scale * np.random.randn(*x.shape)
        x = x - lr * g + noise
        trajectory.append(x.copy())

    return np.array(trajectory)

# Convex: f(x,y) = x² + 4y²
def convex_grad(p):
    return np.array([2*p[0], 8*p[1]])

# Non-convex: f(x,y) = sin(x) + sin(y) + 0.1*(x² + y²)
def nonconvex_grad(p):
    return np.array([np.cos(p[0]) + 0.2*p[0],
                     np.cos(p[1]) + 0.2*p[1]])

# 여러 초기점에서 실행
inits = [np.array([3.0, 3.0]), np.array([-2.0, 3.0]), np.array([0.5, -2.0])]

# Convex: 모두 같은 점으로 수렴
for x0 in inits:
    traj = gradient_descent(convex_grad, x0, lr=0.1)
    print(f"Convex: {x0} → {traj[-1]}")  # 모두 (0, 0) 근처

# Non-convex: 다른 local minima로 수렴
for x0 in inits:
    traj = gradient_descent(nonconvex_grad, x0, lr=0.1)
    print(f"Non-convex: {x0} → {traj[-1]}")  # 서로 다른 점들
```

---

## Challenge 6: Optimizers and Data Geometry in Tiny Transformers

> **책 참조**: Exercise 3.5.1 (Exploring Optimizers and Data Geometry in Tiny Transformers)

### 배경 지식: Adaptive Optimizers

#### SGD, RMSProp, Adam 비교

| Optimizer | 업데이트 규칙 | 핵심 아이디어 |
|-----------|--------------|---------------|
| **SGD** | $\theta \leftarrow \theta - \eta g$ | 단순 기울기 하강 |
| **Momentum** | $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \eta v$ | 관성 추가 |
| **RMSProp** | $s \leftarrow \gamma s + (1-\gamma)g^2$, $\theta \leftarrow \theta - \eta g/\sqrt{s+\epsilon}$ | 기울기 크기로 정규화 |
| **Adam** | 위 둘의 결합 + bias correction | Momentum + RMSProp |

---

### Question 1: 시퀀스 구조가 옵티마이저 성능에 미치는 영향

#### 시퀀스 특성과 Loss Landscape

**반복적 패턴** (예: "abcabcabc"):
```
- 예측 가능한 구조
- Gradient가 일관된 방향
- 수렴 빠름
- SGD도 잘 작동
```

**불규칙 패턴** (예: 자연어):
```
- 복잡한 의존성
- Gradient 방향이 변동적
- Adam의 적응성이 유리
```

**구두점과 공백**:
```
"Hello,world" vs "Hello , world"
→ 토큰화 패턴 변화
→ 다른 위치 임베딩 학습 필요
→ 수렴 특성 변화
```

#### 왜 구조가 중요한가?

Transformer는 attention pattern을 학습:

```
입력: "The cat sat on the mat"
        │   │   │
        └───┴───┴── attention이 학습해야 할 패턴

규칙적 텍스트: 패턴 학습 쉬움
불규칙 텍스트: 다양한 패턴 필요 → 더 많은 iteration
```

---

### Question 2: RMSProp의 정체 현상 vs Adam의 극복

#### RMSProp의 문제: 초기 큰 기울기의 함정

**RMSProp 업데이트**:
$$s_t = \gamma s_{t-1} + (1-\gamma) g_t^2$$
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$$

**문제 시나리오**:
```
t=0: 초기 큰 기울기 g₀ = 100
     s₁ = 0.9×0 + 0.1×100² = 1000

t=1: 작은 기울기 g₁ = 1
     s₂ = 0.9×1000 + 0.1×1 = 900.1

     effective lr = η/√900 = η/30 (매우 작음!)
```

**기하학적 해석**:
```
      Loss
        │
        │╲
        │ ╲  큰 초기 기울기
        │  ╲
        │   ╲______ 여기서 정체
        │          ↑
        │         s가 너무 커서
        │         더 이상 못 내려감
```

#### Adam의 해결책: Bias Correction + 두 개의 moment

**Adam 업데이트**:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2nd moment)}$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t) \quad \text{(bias correction)}$$
$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Adam의 장점**:

1. **Bias Correction**: 초기 추정치 보정
   ```
   t=1일 때 β₁=0.9면:
   m₁ = 0.1×g₁ (과소추정)
   m̂₁ = m₁/(1-0.9) = m₁/0.1 = g₁ (보정됨)
   ```

2. **1st Moment (Momentum)**: 기울기 방향의 관성
   ```
   일관된 방향 → 가속
   진동하는 방향 → 감속
   ```

3. **적응적 스케일링**: 파라미터별로 다른 학습률
   ```
   큰 기울기 파라미터: 작은 effective lr
   작은 기울기 파라미터: 큰 effective lr
   ```

---

### Question 3: Confusion Matrix가 보여주는 것

#### Confusion Matrix 해석

```
           Predicted
           A  B  C  D
        A [90  5  3  2]
Actual  B [ 4 85  8  3]
        C [ 2  6 88  4]
        D [ 1  3  4 92]
```

**대각선**: 올바른 예측 (높을수록 좋음)
**비대각선**: 혼동 패턴

#### Optimization + Data Geometry의 결합 효과

**데이터 기하학이 표현 학습에 미치는 영향**:
```
비슷한 클래스 (예: 'o'와 '0'):
→ 임베딩 공간에서 가깝게 위치
→ 더 많은 혼동
→ confusion matrix의 비대각 원소 큼
```

**옵티마이저의 영향**:
```
SGD: 단순한 결정 경계
     → 간단한 패턴만 학습
     → 특정 클래스 쌍에서 높은 혼동

Adam: 복잡한 결정 경계 가능
      → 미묘한 차이 포착
      → 전반적으로 낮은 혼동
```

**실험에서 확인할 수 있는 것**:
1. 어떤 문자/토큰이 자주 혼동되는가?
2. 학습 초기 vs 후기의 confusion 패턴 변화
3. 옵티마이저에 따른 confusion 패턴 차이

---

## Challenge 7: CNN Capacity, Scale, and Inductive Bias

> **책 참조**: Exercise 4.1.3 (Exploring the Impact of Filter Count and Size in a CNN)

### 배경 지식: CNN 아키텍처

#### Convolutional Layer의 구성 요소

```
Input: H × W × C_in
       │
       ▼
    [Conv2D]  ← K filters of size (F × F × C_in)
       │
       ▼
Output: H' × W' × K
```

**파라미터**:
- **Filter count (K)**: 출력 채널 수
- **Kernel size (F)**: 필터 크기 (F×F)
- **Stride, Padding**: 출력 크기 결정

---

### Question 1: 채널 수 증가의 영향 - 표현력 vs 과적합

#### 표현력 (Expressiveness)

**채널 수 K의 의미**:
```
K = 1: 단 하나의 특징만 탐지
      예: 수직 에지 하나

K = 32: 32가지 다른 특징 탐지
       예: 다양한 방향의 에지, 질감, 패턴

K = 512: 매우 다양한 특징
        예: 복잡한 조합 패턴
```

**MNIST에서의 관찰**:

```
K=4:  간단한 에지만 학습
      → 비슷한 숫자 (5와 6) 혼동
      → 정확도 ~95%

K=32: 충분한 특징 학습
      → 대부분의 숫자 구분
      → 정확도 ~99%

K=128: 불필요하게 많은 특징
       → MNIST는 단순해서 추가 이득 없음
       → 과적합 위험 증가
```

#### 과적합 (Overfitting)

**파라미터 수**:
$$\text{params} = K \times F \times F \times C_{in} + K \text{ (bias)}$$

```
예: F=3, C_in=32
K=32:  32 × 3 × 3 × 32 = 9,216 params
K=128: 128 × 3 × 3 × 32 = 36,864 params
K=512: 512 × 3 × 3 × 32 = 147,456 params
```

**과적합 신호**:
```
Train acc: 100%  ─────────
                          ╲
Test acc:  98%   ───────────────
                   │
                   과적합 gap
```

**MNIST의 특수성**:
- 60,000 학습 샘플
- 상대적으로 간단한 패턴
- 적당한 K (16-64)로 충분

---

### Question 2: 커널 크기와 Receptive Field

#### Receptive Field란?

출력의 한 픽셀이 "보는" 입력 영역의 크기

```
3×3 kernel, 1 layer:  receptive field = 3×3

         [입력 이미지]
         █ █ █
         █ ○ █  ← 3×3 영역이
         █ █ █     하나의 출력 결정

3×3 kernel, 2 layers: receptive field = 5×5

         [입력 이미지]
       █ █ █ █ █
       █ █ █ █ █
       █ █ ○ █ █  ← 5×5 영역이
       █ █ █ █ █     하나의 출력 결정
       █ █ █ █ █
```

#### 큰 커널 vs 작은 커널

**7×7 single layer vs 3×3 three layers**:

```
동일한 receptive field (7×7)

7×7 single:
- params: 7×7×C = 49C
- 한 번에 큰 패턴 포착
- 계산적으로 무거움

3×3 × 3 layers:
- params: 3×(3×3×C) = 27C (더 적음!)
- 비선형성 3번 (더 복잡한 함수 학습)
- 계층적 특징 학습
```

#### 왜 큰 커널이 항상 좋지 않은가?

**1. 파라미터 효율성**:
```
같은 receptive field에서:
- 7×7: 49 params per channel
- 3×3 × 3: 27 params per channel

작은 커널 + 깊이 = 더 효율적
```

**2. 비선형성의 이점**:
```
7×7 한 번: f(Wx)
3×3 세 번: f(W₃f(W₂f(W₁x)))

더 많은 비선형성 = 더 복잡한 함수 근사 가능
```

**3. 지역성 원리**:
```
이미지의 대부분 정보는 지역적:
- 3×3로 작은 패턴 먼저 학습
- 깊은 층에서 조합
- 이것이 계층적 특징 학습의 핵심
```

**MNIST 실험 결과 예측**:
```
3×3 kernel: 가장 효율적 (지역 에지 탐지에 적합)
5×5 kernel: 약간 더 좋거나 비슷
7×7 kernel: 과도함 (28×28 이미지에서)
```

---

### Question 3: 더 깊은 아키텍처와 복잡한 데이터셋

#### MNIST → CIFAR-10 → ImageNet 스케일업

```
MNIST:     28×28×1,   10 classes, 간단한 패턴
CIFAR-10:  32×32×3,   10 classes, 색상+복잡한 패턴
ImageNet:  224×224×3, 1000 classes, 실제 세계 복잡성
```

**필요한 변화**:

1. **채널 수**:
   ```
   MNIST: K=32 충분
   CIFAR: K=64-128 필요
   ImageNet: K=64→256→512→1024 (점진적 증가)
   ```

2. **깊이**:
   ```
   MNIST: 2-3 conv layers
   CIFAR: 10+ layers (VGG, ResNet)
   ImageNet: 50-152 layers (ResNet-50/152)
   ```

3. **Regularization**:
   ```
   더 많은 파라미터 = 더 강한 정규화 필요
   - Dropout
   - Batch Normalization
   - Data Augmentation
   - Weight Decay
   ```

#### 깊은 네트워크의 도전

**Vanishing/Exploding Gradients**:
```
해결: Residual Connections (ResNet)
     y = F(x) + x  ← skip connection

기울기가 shortcut을 통해 직접 전파
```

**계산 비용**:
```
해결:
- Bottleneck layers (1×1 conv로 차원 축소)
- Efficient architectures (MobileNet, EfficientNet)
```

---

## Challenge 8: Compression, Geometry, and Abstraction

> **책 참조**: Exercise 4.3.2 (Exploring Learned Low-Dimensional Manifolds)

### 배경 지식: Representation Learning

#### Deep Network = 점진적 압축

```
Input (784D)
    │
Conv1 → ... → FC → Output (10D)
    │
 점진적으로 정보 압축
 784 → 256 → 128 → 64 → 10
```

**목표**: 분류에 필요한 정보만 보존, 나머지 버림

---

### Question 1: 깊이에 따른 Intrinsic Dimensionality 변화

#### Intrinsic Dimensionality란?

데이터가 실제로 "살고 있는" 차원

```
예: 3D 공간의 종이 (2D manifold)

데이터 점들이 3차원에 있지만
실제로는 2차원 평면 위에 분포
→ intrinsic dim = 2
```

#### PCA로 측정하는 방법

각 층의 activation에 PCA 적용:
$$\text{Energy ratio: } \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2}$$

```
90% 에너지를 설명하는 k = intrinsic dimensionality 추정
```

#### 깊이에 따른 변화 패턴

```
Layer     Intrinsic Dim    해석
─────────────────────────────────────
Input     ~50-100          원본 이미지 복잡도
Conv1     ~30-50           저수준 특징 (에지)
Conv2     ~20-30           중간 특징 (텍스처)
FC1       ~10-15           고수준 추상화
Output    ~10              클래스 수 (압축 완료)
```

#### 계층적 추상화의 의미

```
초기 층: 많은 방향이 중요 (다양한 에지/패턴)
       → 높은 intrinsic dim

깊은 층: 분류에 필요한 방향만 남음
       → 낮은 intrinsic dim

Output: 10개 클래스만 구분
       → intrinsic dim ≈ 10 (또는 그 이하)
```

**직관**:
```
사진 100장 → "고양이냐 개냐"

처음: 모든 픽셀 정보 (높은 차원)
마지막: "고양이/개" 2가지 (1차원!)

중간층들이 점진적으로 정보를 추려냄
```

---

### Question 2: 비선형성이 Data Manifold를 변형하는 방법

#### Linear Transform (비선형성 없음)

$$z = Wx$$

```
타원 모양 분포 → 여전히 타원

○○○           ⬭⬭⬭
○○○  →  W →   ⬭⬭⬭
○○○           ⬭⬭⬭

선형 변환: 회전, 스케일링, 전단만 가능
```

#### ReLU 비선형성

$$a = \text{ReLU}(z) = \max(0, z)$$

```
효과: 음수 부분을 0으로 접음

z 공간:           a 공간:
    │+            │
────┼────   →   ──┼────
   -│             │(접힌 부분)
    │             │

"접기" 연산 → manifold의 위상 변화 가능
```

#### PCA 스펙트럼 비교

```
Before ReLU:              After ReLU:
σ₁ ████████████           σ₁ █████████████████
σ₂ ██████████             σ₂ █████████
σ₃ ████████               σ₃ ████
σ₄ ██████                 σ₄ ██
...                       ...

에너지 집중도 증가!
주요 방향으로 정보 압축
```

**왜 이런 일이 발생하는가?**

1. ReLU가 음수를 0으로 만듦 → 분산 감소
2. 활성화된 뉴런만 정보 전달 → sparse representation
3. 비선형 접기가 선형적으로 분리 불가능한 것을 분리 가능하게

---

### Question 3: 적은 데이터에서 압축 vs 암기

#### 충분한 데이터: Compression (일반화)

```
60,000 학습 샘플

PCA 스펙트럼:
σ₁ █████████████████████
σ₂ ████████████
σ₃ ████████
σ₄ ████
...

→ 급격히 감소 (압축이 잘 됨)
→ 몇 개의 주성분이 대부분 설명
→ 일반화 가능한 특징 학습
```

#### 적은 데이터: Memorization (과적합)

```
1,000 학습 샘플

PCA 스펙트럼:
σ₁ ██████████████████
σ₂ █████████████████
σ₃ ████████████████
σ₄ ███████████████
...

→ 천천히 감소 (압축 안 됨)
→ 많은 방향이 비슷하게 중요
→ 개별 샘플을 "외움"
```

#### 왜 이런 차이가 나는가?

**충분한 데이터**:
```
많은 숫자 "3" 예시:
  ³ ³ ³ ³ ³  (다양한 스타일)

공통점 추출 → "3의 본질"만 인코딩
→ 효율적 압축
→ 낮은 intrinsic dim
```

**적은 데이터**:
```
숫자 "3" 예시 몇 개:
  ³ ³ ³

"이 특정 3들"을 외움
→ 개별 특징까지 저장
→ 높은 intrinsic dim
→ 테스트셋의 다른 "3"에서 실패
```

#### 진단 지표

```
if (test acc << train acc) and (PCA spectrum flat):
    진단: 암기/과적합
    처방: 더 많은 데이터 또는 정규화

if (test acc ≈ train acc) and (PCA spectrum sharp):
    진단: 건강한 일반화
    → 압축이 잘 된 representation
```

---

# Lecture 3-4 Challenges (Ch. 5-6: Probability & Information)

---

## Challenge 9: Learning Transport Maps - Normalizing Flows

> **책 참조**: Exercise 5.2.3 (Experimenting with the 1D Normalizing Flow)

### 배경 지식: Normalizing Flow

#### 핵심 아이디어

복잡한 분포를 간단한 분포로 (또는 역방향으로) 변환하는 학습 가능한 map

```
복잡한 분포 p_Y     ←→     간단한 분포 p_Z (예: N(0,1))
    (Laplace)        g_θ          (Gaussian)

Forward:  Z = g_θ(Y)   (복잡 → 간단)
Inverse:  Y = g_θ⁻¹(Z) (간단 → 복잡, sampling용)
```

#### Change of Variables

$$p_Y(y) = p_Z(g_\theta(y)) \cdot \left|\frac{dg_\theta}{dy}\right|$$

**log-likelihood**:
$$\log p_Y(y) = \log p_Z(g_\theta(y)) + \log \left|\frac{dg_\theta}{dy}\right|$$

---

### Exercise: Laplace → Gaussian Transport

#### Analytic Solution

**Laplace 분포**: $p_Y(y) = \frac{1}{2}e^{-|y|}$
**CDF**: $F_Y(y) = \begin{cases} \frac{1}{2}e^y & y < 0 \\ 1 - \frac{1}{2}e^{-y} & y \geq 0 \end{cases}$

**Optimal Transport Map**:
$$g(y) = \Phi^{-1}(F_Y(y))$$

여기서 $\Phi^{-1}$은 표준 정규분포의 inverse CDF (quantile function)

---

### Question 1: 학습된 map vs Analytic map - 어디서 근사가 깨지는가?

#### 꼬리 영역에서의 문제

```
       y (Laplace)
       │
  0.01 │     ╱────────  analytic g(y)
       │    ╱╱╱╱╱╱╱╱  learned g_θ(y)
       │   ╱
       │  ╱
 -0.01 │─╱
       └─────────────→ z (Gaussian)
         -3  -2  -1  0  1  2  3

         ↑
      꼬리 영역: 차이 발생
```

**이유 1: 데이터 희소성**
```
Laplace 꼬리: |y| > 3인 샘플 < 5%
→ 학습 데이터 부족
→ 신경망이 이 영역을 잘 학습 못함
```

**이유 2: 기울기 폭발**
```
|y| → ∞ 에서:
- F_Y(y) → 0 또는 1
- Φ⁻¹(F_Y(y)) → ±∞
- 기울기 매우 큼 → 불안정
```

**이유 3: 유한 용량**
```
신경망은 유한한 수의 파라미터
→ 모든 영역을 동시에 잘 근사 불가
→ 대부분의 데이터가 있는 중심부 우선 학습
```

---

### Question 2: Heavy-tailed Distribution의 영향

#### Heavy-tail vs Light-tail

```
                      │
                      │
Gaussian (light):   ──┼──  빠르게 감소
                      │
Laplace (medium):   ──┼──  지수 감소
                      │
t-distribution:     ──┼──  천천히 감소 (power law)
(heavy)               │
```

**heavy-tail의 수학적 정의**: $p(y) \propto (1 + y^2)^{-\alpha}$, $\alpha > 1$

#### 학습에 미치는 영향

**1. Training Instability**
```
Heavy-tail → 극단적 샘플 가끔 등장
→ 큰 기울기 스파이크
→ 학습 불안정

해결: Gradient clipping, robust loss
```

**2. Gradient Magnitudes**
```
꼬리에서:
∂loss/∂θ ∝ |g'(y)|
         ∝ p_Z(g(y))/p_Y(y)

heavy-tail p_Y → 작은 분모 → 큰 기울기
```

**3. Map Distortion**
```
Heavy-tail을 Gaussian으로 매핑:
- 꼬리의 많은 확률 질량을
- Gaussian의 좁은 영역으로 압축

→ 극심한 왜곡
→ g'(y)가 매우 가파름
→ 수치적 불안정
```

#### 실험 비교

```python
# α=2 (heavy) vs Laplace (medium) vs Gaussian (light)

# Heavy-tail (Cauchy-like):
# 학습 곡선이 진동, 수렴 느림, 꼬리 근사 실패

# Laplace:
# 적당한 수렴, 꼬리에서 약간의 오차

# Gaussian → Gaussian:
# 가장 쉬움 (identity map 학습)
```

---

## Challenge 10: What Covariance Sees — and What It Misses

> **책 참조**: Exercise 5.3.4 (Empirical Multivariate Statistics)

### 배경 지식: 공분산의 기하학적 의미

#### 공분산 행렬

$$\Sigma = E[(X - \mu)(X - \mu)^T]$$

**기하학적 해석**: 데이터 분포의 "타원" 모양 설명
- 고유벡터: 타원의 주축 방향
- 고유값: 각 축 방향의 퍼짐 정도

```
         y
         │    ⬭  ← Σ로 설명되는 타원
         │   ╱
         │  ╱
    ─────┼────── x
         │
```

---

### Question 1: 왜 공분산이 고차 모멘트보다 빨리 수렴하는가?

#### 모멘트의 정의

- **1차 모멘트** (평균): $E[X]$
- **2차 모멘트** (공분산): $E[XX^T]$
- **3차 모멘트** (skewness): $E[X^3]$
- **4차 모멘트** (kurtosis): $E[X^4]$

#### 수렴 속도 비교

**중심극한정리 (CLT)**:
$$\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)$$

표준오차: $O(1/\sqrt{n})$

**고차 모멘트의 분산**:
$$\text{Var}(\hat{m}_k) \propto E[X^{2k}] - (E[X^k])^2$$

```
k=1 (mean):    Var ∝ E[X²] - μ²
k=2 (variance): Var ∝ E[X⁴] - (E[X²])²
k=3:           Var ∝ E[X⁶] - (E[X³])²
```

**핵심 관찰**: 고차 모멘트 추정에 필요한 분산은 더 높은 차수의 모멘트에 의존

#### 직관적 이해

```
n=100 샘플로 추정:

평균 (k=1):
████████████████████████████████ 정확

분산 (k=2):
████████████████████████ 꽤 정확

Skewness (k=3):
████████████████ 대략적

Kurtosis (k=4):
████████ 불확실

→ 고차 모멘트는 극단값에 민감
→ 극단값은 드물게 관측
→ 더 많은 샘플 필요
```

---

### Question 2: Mixture에서 공분산의 한계

#### Single Gaussian

```
공분산 타원이 분포를 잘 설명:

      │  ⬭⬭⬭
      │ ⬭⬭⬭⬭⬭
      │  ⬭⬭⬭
      └──────────

Σ의 타원 ≈ 실제 분포 모양
```

#### Two-component Mixture

```
      │    ○○○       ○○○
      │   ○○○○○     ○○○○○
      │    ○○○       ○○○
      └──────────────────
         mode 1     mode 2

공분산 타원:
      │  ─────────────────
      │ /                 \
      │/                   \
      └──────────────────────

→ 전체를 하나의 타원으로!
→ Bimodality를 완전히 놓침
```

#### 공분산이 놓치는 것

**올바르게 포착**:
- 전체 평균 위치
- 전체적인 퍼짐 방향
- 두 축 간 상관관계 (대략적)

**놓치는 것**:
- **다봉성 (Multimodality)**: 두 개의 분리된 군집
- **비대칭성**: 왜도 (skewness)
- **Heavy tails**: 첨도 (kurtosis)

#### 왜 공분산만으로 부족한가?

공분산 = **2차 통계량만** 사용

```
Mixture의 특성:
P(X) = π₁N(μ₁, Σ₁) + π₂N(μ₂, Σ₂)

공분산만으로 구분 불가:
- N(0, 2I) (큰 단일 Gaussian)
- 0.5 N(-1, I) + 0.5 N(1, I) (mixture)

둘 다 같은 mean과 covariance!
```

---

### Question 3: Whitening의 한계

#### Whitening이란?

데이터를 변환하여 identity covariance를 갖게 함:
$$Z = \Sigma^{-1/2}(X - \mu)$$

**결과**: $E[Z] = 0$, $\text{Cov}(Z) = I$

#### Single Gaussian에서 Whitening

```
Before:                 After:
    ⬭                     ○
   ⬭⬭⬭                  ○○○
  ⬭⬭⬭⬭⬭    Σ^{-1/2}    ○○○○○
   ⬭⬭⬭      ───→        ○○○
    ⬭                     ○

타원 → 원 (Gaussian을 "정규화")
```

**성공**: Whitening 후 표준 정규분포가 됨

#### Mixture에서 Whitening

```
Before:                 After:
    ○○○       ○○○          ○○○○○○○○○
   ○○○○○     ○○○○○        ○○○○○○○○○○○
    ○○○       ○○○          ○○○○○○○○○
   mode1     mode2
                          여전히 bimodal!

Whitening은 선형 변환
→ 모양만 바꾸고 구조는 유지
→ 두 모드가 여전히 분리
```

#### Heavy-tailed Distribution에서 Whitening

```
Heavy-tail:            After whitening:
     │                      │
     │  ──────             │  ──────  여전히
     │ ╱      ╲            │ ╱      ╲ heavy-tail!
     │╱        ╲           │╱        ╲
    ─┴──────────           ─┴──────────

Whitening은 2차 모멘트만 맞춤
→ 4차 이상 모멘트는 변하지 않음
→ Heavy tail 특성 유지
```

#### 핵심 결론

```
Whitening이 "Gaussianize"하는 조건:
1. 데이터가 이미 Gaussian일 때만!
2. 또는 선형 변환으로 Gaussian이 될 수 있을 때

Whitening이 실패하는 경우:
- Multimodal distributions
- Heavy-tailed distributions
- Any non-Gaussian structure

왜? Whitening은 선형 변환이므로
    비선형적 구조를 바꿀 수 없음
```

---

## Challenge 11: Diagnosing Probabilistic Model Failures

> **책 참조**: Exercise 6.1.2 (Diagnosing Model Failures: Feature Dependence, Calibration, and KL)

### 배경 지식: Naïve Bayes vs Neural Network

#### Naïve Bayes의 가정

$$P(Y|X) \propto P(Y) \prod_{j=1}^{d} P(X_j|Y)$$

**핵심 가정**: 특징들이 클래스가 주어졌을 때 **조건부 독립**

```
P(X₁, X₂ | Y) = P(X₁|Y) × P(X₂|Y)

"클래스를 알면, 각 픽셀은 서로 독립"
```

#### Neural Network

특징 간 의존성을 학습:
- Convolutional layers: 공간적 의존성
- Fully connected layers: 전역적 의존성

---

### Question 1: Naïve Bayes 독립 가정의 위반

#### MNIST에서의 픽셀 의존성

```
숫자 "8":
    ░░██░░
    ░█░░█░
    ░░██░░     픽셀들이 강하게 연관됨:
    ░█░░█░     - 위 동그라미가 있으면
    ░░██░░     - 아래 동그라미도 있음

P(위 동그라미 | "8") × P(아래 동그라미 | "8")
≠ P(위, 아래 동그라미 함께 | "8")
```

#### 혼동되는 숫자 쌍

```
Naïve Bayes가 자주 혼동:
- 4 vs 9: 위쪽 구조 비슷
- 3 vs 8: 곡선 패턴 비슷
- 5 vs 6: 아래 곡선 비슷

이유: 픽셀 조합의 공간적 패턴을 무시
```

#### Neural Network의 해결책

```
Conv layer가 학습하는 것:
- 에지 조합
- 코너 패턴
- 곡선 연결성

"픽셀 A와 B가 이 각도로 연결되어 있으면 3"
→ 조건부 의존성을 명시적으로 모델링
```

---

### Question 2: Accuracy vs Calibration

#### 정의

**Accuracy**: 예측이 맞았는가?
$$\text{Acc} = P(\hat{Y} = Y)$$

**Calibration**: 예측 확률이 실제 확률과 일치하는가?
$$P(Y=k | \hat{P}(Y=k) = p) = p$$

#### 예시: Accurate but Poorly Calibrated

```
모델이 "90% 확률로 고양이"라고 한 100개 샘플:
→ 실제로 85개가 고양이

Accuracy: 85% (괜찮음)
Calibration: 90% 예측에서 85% 실현 (과신)
```

#### 예시: Calibrated but Inaccurate

```
모델이 항상 "50% 고양이, 50% 개"라고 예측:
→ 실제 분포: 50% 고양이, 50% 개

Calibration: 완벽! (50%가 실제로 50%)
Accuracy: 50% (랜덤 추측)
```

#### Calibration Curve

```
     Observed frequency
     │
 1.0 │              ╱ (perfect calibration)
     │            ╱
 0.8 │        ╱  ╱
     │      ╱  ╱   과신 (overconfident)
 0.6 │    ╱  ╱
     │  ╱  ╱
 0.4 │╱  ╱
     │  ╱
 0.2 │╱    과소 신뢰 (underconfident)
     │
     └────────────────→ Predicted probability
       0.2 0.4 0.6 0.8 1.0
```

#### Naïve Bayes vs NN의 Calibration

```
Naïve Bayes:
- 종종 과신 (overconfident)
- 독립 가정 위반 → 확률 곱 과대평가
- 0.99 예측이 실제로는 0.85

Neural Network:
- Cross-entropy 학습 → 자연스러운 calibration
- 하지만 큰 NN은 과신 경향
- Temperature scaling으로 교정 가능
```

---

### Question 3: Per-sample KL Divergence 해석

#### KL Divergence 정의

$$D_{KL}(p_{NN} \| p_{NB}) = \sum_k p_{NN}(k|x) \log \frac{p_{NN}(k|x)}{p_{NB}(k|x)}$$

**의미**: NN의 예측 분포와 NB의 예측 분포가 얼마나 다른가

#### 큰 KL 값이 의미하는 것

```
KL이 큰 샘플:
- NN: "99% 확률로 3"
- NB: "40% 확률로 3, 30% 확률로 8"

→ 두 모델이 구조적으로 다르게 본다
→ 단순히 "둘 다 틀림"이 아님
```

#### Systematic vs Random Disagreement

**Random disagreement** (노이즈):
```
KL 값이 전체적으로 비슷하게 분포
특별한 패턴 없음
```

**Systematic disagreement** (구조적):
```
KL 값이 특정 숫자/패턴에서 높음

예: 모든 "8"에서 KL이 높음
→ NB가 "8"의 두 원 연결을 이해 못함
→ NN은 이 구조를 학습함
```

#### 높은 KL 샘플 분석

```python
# 높은 KL 샘플 찾기
kl_per_sample = compute_kl(p_nn, p_nb)
high_kl_idx = np.argsort(kl_per_sample)[-100:]

# 시각화
for idx in high_kl_idx:
    print(f"NN: {p_nn[idx]}, NB: {p_nb[idx]}")
    show_image(X[idx])

# 패턴 발견:
# - 대부분 구조적으로 복잡한 숫자
# - 픽셀 간 의존성이 중요한 경우
```

---

## Challenge 12: Geometry of Probability — Wasserstein Distance

> **책 참조**: Exercise 6.2.6 (Exploring Wasserstein Geometry)

### 배경 지식: Wasserstein vs KL

#### KL Divergence

$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

**특징**:
- 확률 비율의 기대값
- 비대칭: $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$
- Support가 다르면 ∞

#### Wasserstein Distance (Earth Mover's Distance)

$$W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \int |x - y| d\gamma(x, y)$$

**직관**: "흙더미 P를 Q로 옮기는 최소 비용"

```
P: ████        Q:        ████
   흙더미          목표 위치

W₁ = (흙의 양) × (이동 거리)
```

---

### Question 1: Quantile Function을 통한 W₁ 해석

#### 1D에서의 Wasserstein

$$W_1(P, Q) = \int_0^1 |F_P^{-1}(u) - F_Q^{-1}(u)| du$$

여기서 $F^{-1}$은 quantile function (inverse CDF)

#### 기하학적 의미

```
CDF를 수평으로 볼 때:

P의 CDF    Q의 CDF
    │╱         │  ╱
    │          │ ╱
   ╱│         ╱│
  ╱ │        ╱ │
 ╱  │       ╱  │
────┴───   ───┴───

W₁ = 두 CDF 사이의 면적
```

#### Shift와 Scaling에서의 선형성

**Shift (이동)**:
$$W_1(P, P + c) = |c|$$

"분포를 c만큼 이동하면 W₁도 c만큼"

```
예: N(0,1) → N(3,1)
W₁ = 3 (단순히 평균 차이)
```

**Scaling (스케일링)**:
$$W_1(aP, aQ) = |a| \cdot W_1(P, Q)$$

```
예: N(0,1) vs N(0,4)
두 분포 사이 W₁ ∝ (표준편차 차이)
```

**왜 선형인가?**
```
Quantile로 보면:
F_P^{-1}(u) → F_P^{-1}(u) + c  (shift)
F_P^{-1}(u) → a·F_P^{-1}(u)    (scale)

적분 안에서 선형 변환 → 선형 결과
```

---

### Question 2: Gaussian vs Laplace - KL vs W₁ 비교

#### KL의 문제: Support 불일치

```
Gaussian 꼬리:  e^{-x²/2}  (매우 빠르게 감소)
Laplace 꼬리:   e^{-|x|}   (상대적으로 느리게 감소)

KL 계산 시:
∫ p_L(x) log(p_L(x)/p_G(x)) dx

꼬리에서:
p_L(x)/p_G(x) → ∞  (Laplace >> Gaussian)
→ KL이 크거나 불안정
```

#### W₁의 안정성

```
W₁는 "얼마나 멀리 이동해야 하는가"만 측정
꼬리의 확률 비율이 아님

Laplace → Gaussian 변환:
- 꼬리의 확률 질량을 안쪽으로 이동
- 유한한 이동 거리
- 항상 유한한 값
```

#### 수치 비교

```python
from scipy import stats
import numpy as np

# Gaussian N(0,1) vs Laplace(0, 1/√2) (같은 분산)
gaussian = stats.norm(0, 1)
laplace = stats.laplace(0, 1/np.sqrt(2))

# 샘플 기반 추정
n = 10000
g_samples = gaussian.rvs(n)
l_samples = laplace.rvs(n)

# W₁: 정렬 후 차이의 평균
g_sorted = np.sort(g_samples)
l_sorted = np.sort(l_samples)
W1 = np.mean(np.abs(g_sorted - l_sorted))
print(f"W₁ ≈ {W1:.3f}")  # 안정적인 값

# KL: 밀도 추정 필요, 꼬리에서 불안정
# (직접 계산 시 수치적 문제 발생)
```

---

### Question 3: Neural Transport Map으로 W₁ 최소화

#### Kantorovich-Rubinstein Duality

$$W_1(P, Q) = \sup_{\|f\|_L \leq 1} E_{x \sim P}[f(x)] - E_{y \sim Q}[f(y)]$$

**의미**: 1-Lipschitz 함수들 중에서 두 분포의 기대값 차이를 최대화

#### Neural Transport Approach

```
학습 목표:
T_θ: P → Q such that T_θ(X) ~ Q when X ~ P

손실 함수:
L(θ) = W₁(T_θ#P, Q)
     ≈ (정렬된 샘플 간 차이의 평균)
```

#### Analytic Optimal Transport와 비교

**1D Gaussian → Gaussian**:
```
Analytic: T*(x) = σ_Q/σ_P · (x - μ_P) + μ_Q
(선형 변환)

Neural: T_θ(x) ≈ ax + b를 학습
→ 충분히 학습하면 analytic에 수렴
```

#### 학습된 Map의 특성

```
장점:
- 복잡한 분포에도 적용 가능
- Analytic solution이 없을 때 유용

단점:
- 수렴 보장 없음
- 고차원에서 어려움
- Lipschitz 제약 강제 필요 (gradient penalty 등)
```

#### Python 코드 스케치

```python
import torch
import torch.nn as nn

class TransportNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

def wasserstein_loss(transported, target):
    """정렬 기반 W₁ 근사"""
    t_sorted = torch.sort(transported.squeeze())[0]
    q_sorted = torch.sort(target.squeeze())[0]
    return torch.mean(torch.abs(t_sorted - q_sorted))

# 학습
transport = TransportNet()
optimizer = torch.optim.Adam(transport.parameters(), lr=1e-3)

for epoch in range(1000):
    p_samples = torch.randn(256, 1)  # source: N(0,1)
    q_samples = torch.randn(256, 1) * 2 + 3  # target: N(3,4)

    transported = transport(p_samples)
    loss = wasserstein_loss(transported, q_samples)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습된 map 확인
# 이상적으로: T(x) ≈ 2x + 3
```

---

# 요약: 12개 Challenge의 핵심 교훈

| Challenge | 핵심 개념 | AI 응용 |
|-----------|----------|---------|
| **1** | Einstein Notation | GPU 연산 효율성 |
| **2** | SVD/Low-rank | 차원 축소, 압축 |
| **3** | Backpropagation | 신경망 학습의 기초 |
| **4** | ODE Regression | 물리 기반 AI |
| **5** | Convex/Non-convex | 최적화 지형 이해 |
| **6** | Adaptive Optimizers | Transformer 학습 |
| **7** | CNN Capacity | 아키텍처 설계 |
| **8** | Representation Learning | 특징 압축 |
| **9** | Normalizing Flows | 생성 모델 |
| **10** | Covariance Limits | 분포 모델링의 한계 |
| **11** | Model Diagnostics | 확률 모델 비교 |
| **12** | Wasserstein Geometry | 분포 간 거리 |

---

# 참고 자료

- **책**: Mathematics of Generative AI (MathGenAIBook12_14_25.pdf)
- **기존 해설**: [notebook/chapter1/KAIST-challenges-solutions.md](../chapter1/KAIST-challenges-solutions.md)
- **관련 노트북**:
  - [chapter3/](../chapter3/) - Optimization notebooks
  - [chapter4/](../chapter4/) - Deep Learning notebooks
  - [chapter5/Statistics/](../chapter5/Statistics/) - Probability notebooks
  - [chapter6/Info/](../chapter6/Info/) - Information Theory notebooks
