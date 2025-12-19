# Chapter 2: Calculus and Differential Equations (in AI)

> **책 페이지**: 38-63
> **핵심 주제**: Automatic Differentiation, ODE 기초, 선형 시스템, 그래프 역학
> **KAIST Challenge 연결**: Challenge 3 (Chain Rule & Backprop), Challenge 4 (ODE 회귀)

---

## 📚 목차

1. [Automatic Differentiation (AD)](#1-automatic-differentiation-ad)
2. [Ordinary Differential Equations (ODEs)](#2-ordinary-differential-equations-odes)
3. [System of Linear ODEs](#3-system-of-linear-odes)
4. [Notebooks 가이드](#4-notebooks-가이드)
5. [Generative AI에서의 응용](#5-generative-ai에서의-응용)

---

## 1. Automatic Differentiation (AD)

### 미분이 뭐였더라? (복습)

고등학교 때 배운 미분을 떠올려봅시다:

```
f(x) = x²

"x가 조금 변하면, f(x)는 얼마나 변할까?"

x = 3일 때:
f(3) = 9
f(3.001) = 9.006001

변화량 = 0.006001 / 0.001 ≈ 6

이게 바로 f'(3) = 2×3 = 6 (미분값)
```

**미분 = "변화율"** = "입력이 조금 바뀌면 출력이 얼마나 바뀌나?"

### 왜 AI에서 미분이 필요한가?

```
신경망 학습의 핵심 질문:

"Loss가 10이었는데, 이걸 줄이려면
 가중치 W를 어떻게 바꿔야 할까?"

답: ∂Loss/∂W를 계산해서
    Loss가 줄어드는 방향으로 W를 조정!

이게 바로 Gradient Descent (경사하강법)
```

### 문제: 신경망은 함수가 엄청 복잡함

```
단순 함수:     f(x) = x²           → 미분 쉬움: 2x

신경망:        f(x) = σ(W₃ · σ(W₂ · σ(W₁ · x)))
               ↑ 함수 안에 함수 안에 함수...

손으로 미분? → 불가능!
```

### AD = 컴퓨터가 자동으로 미분해주는 것

> **책 원문 (p.38):**
> "Automatic differentiation (AD) is the backbone of modern scientific computing and machine learning."

**핵심 아이디어**:
아무리 복잡한 함수도 **기본 연산의 조합**이다!

```
f(x₁, x₂) = sin(x₁ + x₂) / x₁

이걸 분해하면:
① 더하기:  x₁ + x₂
② sin:    sin(①)
③ 나누기:  ② / x₁

각 기본 연산의 미분은 알고 있으니까,
Chain Rule로 연결하면 전체 미분을 구할 수 있다!
```

### Chain Rule = 미분의 연쇄 법칙

고등학교 때 배운 합성함수 미분:

```
h(x) = f(g(x)) 일 때

h'(x) = f'(g(x)) × g'(x)

예시: h(x) = sin(x²)
      g(x) = x²,  f(u) = sin(u)

      h'(x) = cos(x²) × 2x
```

**직관적 이해**:
```
x가 변하면 → g가 변하고 → f가 변한다

"x가 1 변할 때 g가 3 변하고,
 g가 1 변할 때 f가 2 변하면,
 x가 1 변할 때 f는 3×2=6 변한다"
```

### 미분 계산 방법 3가지 비교

| 방법 | 어떻게? | 문제점 |
|------|---------|--------|
| **손 계산** | 수식 전개 | 복잡하면 불가능 |
| **수치 미분** | $\frac{f(x+0.0001) - f(x)}{0.0001}$ | 오차 있음, 느림 |
| **AD** | 기본 연산 분해 + Chain Rule | ✅ 정확하고 빠름 |

### Forward Mode vs Reverse Mode

**Forward Mode** (순방향):
```
질문: "x₁이 변하면 출력이 얼마나 변할까?"

x₁ → [연산1] → [연산2] → [연산3] → 출력
 ↓       ↓         ↓         ↓        ↓
 1    →  ?    →    ?    →    ?   →   최종 미분값

앞에서부터 변화를 추적해나감
```

**Reverse Mode** (역방향 = Backpropagation):
```
질문: "출력(Loss)이 1 변할 때, 각 입력은 얼마나 책임이 있나?"

x₁ → [연산1] → [연산2] → [연산3] → Loss
 ↑       ↑         ↑         ↑        ↓
 ?   ←   ?    ←    ?    ←    ?   ←    1

뒤에서부터 책임을 추적해나감
```

### 언제 뭘 쓰나?

```
Forward:  입력 적고, 출력 많을 때
          예: "x가 변하면 y₁, y₂, ..., y₁₀₀₀ 각각 얼마나 변하나?"

Reverse:  입력 많고, 출력 적을 때 (대부분의 AI!)
          예: "Loss 하나를 줄이려면 W₁, W₂, ..., W₁₀억 각각 어떻게?"
```

> **책 원문 (p.40):**
> "Forward mode cost ~ O(n), Reverse mode cost ~ O(m)"
> (n = 입력 수, m = 출력 수)

**신경망 학습**:
- 출력 = Loss 1개
- 입력 = 파라미터 수백만~수십억 개
- → **Reverse Mode (Backpropagation)** 가 압도적으로 효율적!
- **Many-to-One vs Many-to-Many**:
    - **Many-to-One**: 하나의 Loss를 최적화하기 위해 수백만 개의 파라미터를 미분하는 일반적인 AI 학습에 필수적입니다.
    - **Many-to-Many**: 일부 생성형 AI 작업에서는 출력도 다수일 수 있으며, AD를 특정 작업에 맞춰 튜닝하기도 합니다.
- **Adjoint Method**: 미분방정식(ODE) 시스템에서 미분을 효율적으로 계산하기 위한 핵심 도구로 언급되었습니다.

### 비유: 책임 추적

```
시험 점수가 나빴다 (Loss가 크다)
    ↓ 왜?
수학 문제를 틀렸다 (출력층 오차)
    ↓ 왜?
공식을 잘못 기억했다 (은닉층 오차)
    ↓ 왜?
수업 시간에 졸았다 (입력층 = 가중치 문제)

Backpropagation = "최종 결과에서 거꾸로 책임 추적"
각 단계가 최종 실패에 얼마나 기여했는지 계산
```

### PyTorch에서는 어떻게?

```python
import torch

# 1. 텐서 만들고, "미분 추적해줘" 설정
x = torch.tensor([2.0], requires_grad=True)

# 2. 복잡한 연산 수행
y = x**2 + 3*x + 1  # y = x² + 3x + 1

# 3. 역방향 미분 (Backprop)
y.backward()

# 4. 결과 확인
print(x.grad)  # dy/dx = 2x + 3 = 2(2) + 3 = 7
```

PyTorch가 내부적으로 **Computational Graph**를 만들어서 자동으로 미분!

---

## 2. Ordinary Differential Equations (ODEs)

### 미분방정식이 뭔가요?

**보통 방정식** (중학교):
```
2x + 3 = 7  →  x = 2 (숫자를 찾음)
```

**미분방정식**:
```
dx/dt = 2x  →  x(t) = ? (함수를 찾음)

"x의 변화율이 x의 2배다"
→ 이 조건을 만족하는 함수 x(t)는 뭘까?
```

### 실생활 예시로 이해하기

**예시 1: 은행 이자 (복리)**
```
"잔고의 증가율 = 잔고의 5%"

수식으로: dx/dt = 0.05 × x

해석: 돈이 많을수록 더 빨리 늘어남
해:   x(t) = x₀ × e^(0.05t)  (지수 성장)
```

**예시 2: 커피 식기**
```
"온도 변화율 = -(현재온도 - 실온)"

수식으로: dT/dt = -k(T - 20)

해석: 뜨거울수록 빨리 식고, 실온에 가까워지면 천천히 식음
```

**예시 3: 인구 증가**
```
"인구 증가율 = 현재 인구에 비례"

수식으로: dN/dt = rN

해석: 인구가 많을수록 더 빨리 늘어남 (지수 성장)
```

### 표기법 정리

```
dx/dt  =  ẋ  =  x'  (다 같은 표현!)

"x를 t에 대해 미분한 것"
"x의 시간에 따른 변화율"
"x가 시간당 얼마나 변하는가"
```

### ODE의 일반 형태

> **책 원문 (p.44):**
> "An Ordinary Differential Equation (ODE) is an equation involving derivatives of a function with respect to a single independent variable."

$$\frac{dx}{dt} = f(x, t)$$

**해석**: "x의 변화율은 현재 x값과 시간 t에 의해 결정된다"

### 간단한 ODE 풀어보기

**문제**: $\frac{dx}{dt} = 2x$, 초기값 $x(0) = 3$

```
해석: "x의 변화율이 x의 2배"

직관: x가 크면 → 변화율도 큼 → x가 더 빨리 커짐
     → 지수적 성장!

풀이: dx/x = 2dt
      ln|x| = 2t + C
      x = e^(2t+C) = Ae^(2t)

초기조건: x(0) = 3 → A = 3

답: x(t) = 3e^(2t)
```

### 고정점 (Equilibrium) = 변화가 멈추는 곳

```
dx/dt = 0이 되는 x값

예: dx/dt = x(1-x) 에서
    x=0 또는 x=1 이면 변화율이 0
    → 그 상태에서 영원히 머무름
```

**AI에서의 의미**:
```
Loss의 기울기가 0인 점 = 고정점
= 학습이 멈추는 곳
= (희망적으로) 최적의 파라미터!
```

### 2차 ODE: 진동하는 시스템

**스프링에 매달린 물체**:
```
힘 = 질량 × 가속도
-kx = m × d²x/dt²

정리: d²x/dt² = -(k/m)x

"위치의 가속도가 위치에 비례하고 반대 방향"
→ 왔다갔다 진동!
```

**감쇠 진동자** (마찰이 있는 스프링):
$$\frac{d^2x}{dt^2} + \gamma \frac{dx}{dt} + \omega^2 x = 0$$

```
γ = 마찰 계수 (클수록 빨리 멈춤)
ω = 고유 진동수 (스프링 강성)

γ 작음: 오래 진동하다가 멈춤 (under-damped)
γ 큼:   진동 없이 천천히 원점으로 (over-damped)
```

### Double-Well Potential: AI와의 연결

```
포텐셜 에너지:  U(x) = x⁴/4 - x²/2

모양:
        U
        │    *
        │   * *
        │  *   *
        │ *     *        ← 두 개의 "우물"
        │*   *   *
        └──────────→ x
           -1  0  +1

x = -1과 x = +1에서 안정 (공이 머무는 곳)
x = 0은 불안정 (언덕 꼭대기)
```

**AI에서의 의미**:
```
- 두 우물 = 두 개의 클래스 (이진 분류)
- 공의 위치 = 모델의 예측
- 감쇠 = 학습 과정

초기 위치에 따라 어느 우물에 빠질지 결정
= 초기화에 따라 어느 클래스로 분류할지 결정
```

### ODE를 왜 배우나? → Neural ODE, Diffusion Model!

여기서부터 핵심입니다. 천천히 따라와 주세요.

---

#### Step 1: 먼저 ResNet이 뭔지 알아야 해요

**문제 상황**: 신경망을 깊게 쌓으면 성능이 좋아질까?

```
얕은 신경망:  입력 → [층1] → [층2] → 출력    (2층)
깊은 신경망:  입력 → [층1] → [층2] → ... → [층100] → 출력

직관적으로: 층이 많으면 더 복잡한 걸 배울 수 있을 것 같죠?

현실: 층을 너무 깊게 쌓으면 오히려 학습이 안 됨!
     (gradient vanishing 문제)
```

**ResNet의 아이디어** (2015년, 혁명적):

```
기존 방식:
  h₂ = f(h₁)      ← "h₁을 완전히 새로운 h₂로 변환"

ResNet 방식:
  h₂ = h₁ + f(h₁)  ← "h₁에다가 작은 변화 f(h₁)만 더하자"
       ↑
       이 부분이 핵심!
```

**비유로 이해하기**:

```
기존 방식 = "매 층마다 완전히 새로 그리기"
  1층: 백지에서 그림 그림
  2층: 다시 백지에서 그림 그림
  3층: 또 다시 백지에서...
  → 이전에 배운 게 사라질 수 있음

ResNet = "이전 그림 위에 덧칠하기"
  1층: 백지에서 그림 그림
  2층: 1층 그림 + 수정사항
  3층: 2층 그림 + 수정사항
  → 이전 정보가 그대로 전달됨!
```

**수식으로 정리**:

$$h_1 = \text{입력}$$
$$h_2 = h_1 + f(h_1) \quad \leftarrow \text{1층 출력 + 변화량}$$
$$h_3 = h_2 + f(h_2) \quad \leftarrow \text{2층 출력 + 변화량}$$

---

#### Step 2: 이게 ODE랑 무슨 관계?

자, 여기서 마법이 일어납니다.

**Euler method 복습** (ODE 푸는 가장 단순한 방법):

ODE: $\frac{dx}{dt} = f(x)$ → "x의 변화율이 f(x)다"

이걸 컴퓨터로 풀려면? 시간을 잘게 쪼개서:

$$x(t+\Delta t) \approx x(t) + f(x(t)) \cdot \Delta t$$

해석: "다음 x = 지금 x + 변화율 × 시간간격"

**$\Delta t = 1$로 두면?**:

$$x(t+1) = x(t) + f(x(t))$$

**잠깐, 이거 어디서 봤지?**:

| 모델 | 수식 |
|------|------|
| ResNet | $h_{t+1} = h_t + f(h_t)$ |
| Euler | $x_{t+1} = x_t + f(x_t)$ |

→ **완전히 똑같은 형태!**

**결론**:
- ResNet의 각 층을 지나는 것 = ODE를 Euler method로 한 스텝 적분하는 것!
- 층 1개 통과 = 시간 1만큼 흐른 것
- 층 100개 통과 = 시간 100만큼 흐른 것

---

#### Step 3: Neural ODE란?

ResNet이 ODE의 이산화라면... 거꾸로 생각해볼까요?

**ResNet (이산적)**:
$$h_0 \to h_1 \to h_2 \to h_3 \to \cdots \to h_{100}$$
("100개의 층을 차례대로 통과")

**Neural ODE (연속적)**:
$$h(0) \xrightarrow{\frac{dh}{dt} = f(h, t; \theta)} h(T)$$
("0부터 T까지 연속적으로 변환")

**비유**:
- ResNet = 계단으로 100층 올라가기 (한 층, 한 층...)
- Neural ODE = 엘리베이터로 올라가기 (연속적으로 스무스하게)

**Neural ODE의 수식**:

$$\frac{dh}{dt} = f(h, t; \theta)$$

| 기호 | 의미 |
|------|------|
| $h$ | 데이터의 현재 상태 (이미지, 텍스트의 표현) |
| $t$ | "시간" (신경망에서의 깊이) |
| $f$ | 신경망이 학습하는 함수 |
| $\theta$ | 신경망의 가중치 |

→ "데이터 $h$가 시간 $t$에 따라 어떻게 변하는지를 신경망 $f$로 학습한다"

**왜 이렇게 하나?**:

| 장점 | ResNet | Neural ODE |
|------|--------|------------|
| 메모리 | $h_1, h_2, \ldots, h_{100}$ 모두 저장 | $h(0)$과 $h(T)$만 저장 |
| 깊이 | 층 수 미리 고정 (50층? 100층?) | 적분 시간 $T$ 조절 |
| 해석 | 이산적 변환 | 연속적 흐름으로 이해 가능 |

---

#### Step 4: Diffusion Model과 ODE

이제 가장 흥미로운 부분!

**Diffusion Model = "노이즈에서 이미지를 만드는 모델"**
- 사용 예시: DALL-E, Midjourney, Stable Diffusion

**핵심 아이디어 - Forward Process** (이미지 망가뜨리기):

$$x_0 \xrightarrow{t=1} x_1 \xrightarrow{t=2} x_2 \xrightarrow{\cdots} x_T \sim \mathcal{N}(0, I)$$

(마치 잉크를 물에 떨어뜨리면 점점 퍼지는 것처럼!)

**이건 SDE로 표현됨!**:

$$dx = f(x,t)dt + g(t)dW$$

| 항 | 의미 |
|------|------|
| $dx$ | $x$의 작은 변화 |
| $f(x,t)dt$ | 결정적인 변화 (drift) |
| $g(t)dW$ | 랜덤한 노이즈 추가 (diffusion) |

→ "이미지에 체계적으로 노이즈를 더해가는 과정"

**Reverse Process** (이미지 복원하기):

$$x_T \xrightarrow{t=T-1} \cdots \xrightarrow{t=1} x_1 \xrightarrow{t=0} x_0$$

Forward를 거꾸로 돌리면 노이즈에서 이미지가 나옴!

**Reverse도 ODE로 표현됨!**:

$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p(x)\right]dt$$

여기서 $\nabla_x \log p(x)$ = **score function** = 신경망이 학습하는 부분!

→ "각 시점에서 노이즈를 어느 방향으로 빼야 원래 이미지 쪽으로 갈까?"를 배움

**전체 그림**:

| 단계 | 과정 |
|------|------|
| **학습** | 진짜 이미지 $\xrightarrow{\text{Forward}}$ 노이즈 → 신경망이 "노이즈 제거 방향" 학습 |
| **생성** | 랜덤 노이즈 $\xrightarrow{\text{Reverse}}$ 새 이미지! |

**비유**:
- Forward = 조각상에 흙을 덮어서 숨기기 (다비드상 → 흙더미)
- Reverse = 흙을 조금씩 걷어내기 (흙더미 → 새로운 조각상!)
- 신경망 = "어디를 파야 조각상이 나올까?" 학습

---

#### 정리: ODE가 AI에서 왜 중요한가?

**핵심 방정식**: $\frac{dx}{dt} = f(x)$ ("변화율을 기술하는 방정식")

| 모델 | 수식 | 해석 |
|------|------|------|
| **ResNet** | $h_{t+1} = h_t + f(h_t)$ | 이산화된 ODE |
| **Neural ODE** | $\frac{dh}{dt} = f(h; \theta)$ | 연속적인 신경망 |
| **Diffusion** | $\frac{dx}{dt} = f(x, t)$ | 노이즈↔이미지 변환 |

**한 문장 요약**:
> ODE는 "시간에 따른 변화"를 다루는 수학이고,
> 신경망의 "층을 통과하는 변화"나
> Diffusion의 "노이즈 추가/제거 과정"이
> 모두 ODE로 표현된다!

---

## 3. System of Linear ODEs

### 여러 변수가 서로 영향을 주면?

**예시: 토끼와 여우** (Lotka-Volterra 방정식)

$$\frac{dx}{dt} = ax - bxy \quad \text{(토끼: 번식 - 여우한테 잡아먹힘)}$$
$$\frac{dy}{dt} = cxy - dy \quad \text{(여우: 토끼 먹고 번식 - 자연사)}$$

- 토끼가 많으면 → 여우가 잘 먹고 번식
- 여우가 많으면 → 토끼가 줄어듦
- → 주기적으로 오르락내리락!

### 선형 시스템의 경우

$$\frac{d\mathbf{x}}{dt} = A\mathbf{x}$$

- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ (n개의 변수)
- $A$ = n×n 행렬 (변수들이 서로 어떻게 영향주는지)

**예시**:
$$\frac{dx_1}{dt} = 2x_1 - x_2, \quad \frac{dx_2}{dt} = x_1 + 3x_2$$

행렬로:
$$\frac{d}{dt}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

### 해: 행렬 지수 함수

| 차원 | ODE | 해 |
|------|-----|-----|
| 1차원 | $\frac{dx}{dt} = ax$ | $x(t) = e^{at} \cdot x_0$ |
| n차원 | $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$ | $\mathbf{x}(t) = e^{At} \cdot \mathbf{x}_0$ |

**행렬 지수란?**
$$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$

숫자의 지수함수 정의를 행렬로 확장한 것!

### 고유값이 시스템의 운명을 결정

$A$의 고유값 $\lambda$가:

| 고유값 | 의미 | 동작 |
|--------|------|------|
| $\lambda < 0$ (음수) | 안정 | 그 방향으로 수축 |
| $\lambda > 0$ (양수) | 불안정 | 그 방향으로 발산 |
| $\lambda = ai$ (허수) | 진동 | 그 방향으로 회전 |

→ 모든 고유값의 실수부가 음수 → 시스템이 원점으로 수렴 (안정)

### 열 확산 예시

1차원 막대의 온도 분포:
- 점 1: 뜨거움 / 점 2: 미지근 / 점 3: 차가움

각 점은 이웃과 온도를 교환:
$$\frac{dT_i}{dt} = T_{i-1} + T_{i+1} - 2T_i$$

시간이 지나면? → 모든 점이 평균 온도로 수렴!

### Graph Dynamics: SNS에서의 의견 전파

5명이 서로 연결되어 있고, 각자 의견(숫자)을 가지고 있음.
연결된 친구들의 의견 평균 쪽으로 조금씩 이동:

$$\frac{dx_i}{dt} = \sum_{j \in \mathcal{N}(i)} (x_j - x_i)$$

시간이 지나면? → 모두 같은 의견으로 수렴! (Consensus)

**Graph Laplacian** $L = D - A$:

| 기호 | 의미 |
|------|------|
| $A$ | 인접 행렬 (누가 연결됐나) |
| $D$ | 차수 행렬 (각 노드의 연결 수) |
| $L$ | 그래프의 "확산 연산자" |

$$\frac{d\mathbf{x}}{dt} = -L\mathbf{x} \quad \rightarrow \quad \text{의견이 확산되어 평균으로 수렴}$$

---

## 4. Notebooks 가이드

### AD/ 폴더

| 노트북 | 뭘 배우나? | 실습 내용 |
|--------|-----------|-----------|
| `AD.ipynb` | AD의 원리 | PyTorch로 미분 계산, 수치미분과 속도 비교 |

**직접 해볼 것**:
```python
# 복잡한 함수도 AD로 한방에!
x = torch.tensor([2.0], requires_grad=True)
y = torch.sin(x) * torch.exp(-x**2)
y.backward()
print(x.grad)  # 손으로 계산하기 힘든 미분값이 바로!
```

### ODEs/ 폴더

| 노트북 | 뭘 배우나? | 실습 내용 |
|--------|-----------|-----------|
| `linear_systems.ipynb` | 열 확산 | 막대의 온도가 퍼지는 과정 |
| `graph_dynamics.ipynb` | 의견 수렴 | 그래프에서 consensus 도달 |
| `timeordered.ipynb` | 시간 의존 시스템 | A(t)가 변할 때의 복잡성 |

### regression/ 폴더

| 노트북 | 뭘 배우나? | 실습 내용 |
|--------|-----------|-----------|
| `regression.ipynb` | ODE 파라미터 추정 | 노이즈 데이터에서 γ 찾기 |
| `double-well.ipynb` | 이중 우물 역학 | 초기값에 따른 분기 |

### 💡 Lecture 2 Recap: ODE 기반 회귀의 강점
Misha 교수는 강의에서 **데이터 기반(Polynomial) 회귀**와 **ODE 기반 회귀**를 직접 비교했습니다.

1. **다항식 회귀(Polynomial Regression)**:
    - 고차 다항식을 쓰면 데이터 사이의 간격(Gap)에서 요동치거나, 샘플링 범위 밖(Extrapolation)에서 예측이 완전히 어긋나는 과적합(Overfitting) 문제가 발생합니다.
2. **ODE 기반 복원(ODE-based Reconstruction)**:
    - 프로세스가 ODE(예: $\dot{x} = -\gamma x + \text{noise}$)를 따른다는 물리적 지식이 있다면, 단 하나 또는 소수의 파라미터($\gamma$)만으로도 전체 흐름을 정확하게 복원할 수 있습니다.
    - 노이즈에 훨씬 강력하며, 데이터가 없는 구간에서도 물리 법칙에 따라 합리적인 예측이 가능합니다.

> [!TIP]
> **"Physics/Model as a Regularizer"**: 물리 법칙 자체가 강력한 정규화(Regularization) 역할을 수행하여, 더 적은 데이터로도 더 정확한 학습이 가능해집니다.

---

## 5. Generative AI에서의 응용

### AD → 모든 딥러닝의 기반

```
PyTorch, TensorFlow, JAX...
전부 AD 엔진이 핵심!

loss.backward()  ← 이 한 줄이
                   수십억 파라미터의 미분을
                   자동으로 계산해줌

AD 없이는 현대 AI가 불가능!
```

### ODE → Neural ODE, Diffusion Model

**Neural ODE**:
```
기존: 층을 쌓는다 (h₁ → h₂ → h₃ → ...)
새로운 관점: 연속 변환 (dh/dt = f(h))

장점:
- 메모리 효율적 (중간 상태 저장 불필요)
- 불규칙 시계열에 적합
- 물리적 해석 가능
```

**Diffusion Model**:
```
Forward: 이미지 → 노이즈 (점점 더해감)
Reverse: 노이즈 → 이미지 (ODE/SDE로 되돌림)

핵심 수식:
dx = f(x,t)dt + g(t)dW  (SDE)

"노이즈 제거 과정"을 미분방정식으로 모델링!
```

### Graph ODE → GNN의 이론적 기반

**Graph Neural Network의 메시지 전달**:

$$h_i^{(t+1)} = h_i^{(t)} + \sum_{j \in \mathcal{N}(i)} f(h_i, h_j)$$

**이건 ODE의 Euler discretization!**

$$\frac{dh_i}{dt} = \sum_{j \in \mathcal{N}(i)} f(h_i, h_j)$$

| 기호 | 의미 |
|------|------|
| $h_i^{(t)}$ | 노드 $i$의 $t$번째 층 표현 |
| $\mathcal{N}(i)$ | 노드 $i$의 이웃 집합 |
| $f(h_i, h_j)$ | 노드 $i$와 이웃 $j$의 상호작용 함수 |

**연결**:
- GNN 한 층 통과 = ODE를 $\Delta t = 1$로 적분
- GNN 층을 깊게 쌓는 것 = ODE를 오래 적분하는 것

---

## 📝 핵심 정리

### 이 챕터에서 꼭 기억할 것

1. **AD = 복잡한 함수의 미분을 자동으로**
   - Reverse Mode = Backpropagation
   - 신경망 학습의 핵심 엔진

2. **ODE = "변화율"을 기술하는 방정식**
   - 현재 상태가 미래를 결정
   - 고정점 = 변화가 멈추는 곳

3. **선형 시스템의 해 = 행렬 지수**
   - 고유값이 안정성 결정
   - 그래프에서의 확산/수렴

4. **AI 연결**
   - AD → 모든 학습
   - ODE → Neural ODE, Diffusion
   - Graph ODE → GNN

---

## 🔗 다른 챕터와의 연결

| 연결 | 설명 |
|------|------|
| **Ch.1 → Ch.2** | 행렬 지수에서 고유값 분해(ED) 사용 |
| **Ch.2 → Ch.3** | GD = ODE의 이산화: $\dot{W} = -\nabla L$ |
| **Ch.2 → Ch.4** | **ResNet & Skip Connections**: $h_{t+1} = h_t + f(h_t)$ 식은 사실 **ODE의 Euler Discretization**($\dot{h} = f(h)$)과 정확히 동일합니다. 즉, 깊은 신경망은 미분방정식을 오랫동안 적분하는 과정으로 이해할 수 있습니다. |
| **Ch.2 → Ch.7** | ODE → SDE (확률 추가) → Diffusion Model |

---

*이 문서는 Mathematics of Generative AI Book Chapter 2의 학습 가이드입니다.*
