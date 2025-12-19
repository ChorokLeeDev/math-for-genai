# KAIST AI Mini-Course Challenge 완벽 해설

> **목표**: 기본 개념부터 차근차근 설명하여 누구나 이해할 수 있는 상세 답지
>
> **참조 교재**: Mathematics of Generative AI (MathGenAIBook12_14_25.tex)
>
> **Chapter 1 README**: [README.md](README.md)

---

# 책-Challenge 매핑 테이블

| Challenge | 책 Exercise | 책 페이지 | 핵심 개념 |
|-----------|------------|-----------|-----------|
| **1** | Exercise 1.1.3 (Einstein Summation) | Ch.1 | 텐서 축약, GPU 효율성 |
| **2** | Exercise 1.2.2 (SVD for Matrix Completion) | Ch.1 | Low-Rank Approximation |
| **3** | Exercise 2.1.1 (AD and Jacobian) | Ch.2 | Chain Rule, Backprop |
| **4** | Exercise 2.2.3 (ODE Regression) | Ch.2 | 파라미터 추정 |
| **5** | Exercise 3.2.1 (GD Trajectories) | Ch.3 | Convex/Non-Convex |
| **6** | Exercise 3.5.1 (Tiny Transformers) | Ch.3 | SGD, Adam, RMSProp |
| **7** | Exercise 4.1.3 (CNN Filter/Size) | Ch.4 | CNN Capacity |
| **8** | Exercise 4.3.2 (PCA on Activations) | Ch.4 | Representation Learning |

---

# Lecture 1 Challenges (Ch. 1-2: Linear Algebra & Calculus)

---

## Challenge 1: 텐서 축약과 Einstein Notation

> **책 참조**: Exercise 1.1.3 (Einstein Summation and Computational Efficiency)

### 배경 지식: Einstein Summation Convention이란?

#### 왜 Einstein Notation을 쓰는가?

**문제점**: 행렬/텐서 연산을 수식으로 쓰면 Σ(시그마)가 너무 많아짐

```
일반 표기: y_i = Σ_j A_ij x_j  (합의 기호가 복잡)
Einstein:  y_i = A_ij x_j      (반복 인덱스는 자동으로 합산)
```

**규칙**: 같은 인덱스가 두 번 나오면 → 그 인덱스에 대해 합산

```
예시 1: A_ij x_j  → j가 두 번 나옴 → Σ_j A_ij x_j
예시 2: A_ii      → i가 두 번 나옴 → Σ_i A_ii = trace(A)
```

#### 인덱스의 종류

| 종류 | 설명 | 예시 |
|------|------|------|
| **Free index** | 한 번만 나타남 → 결과에 남음 | y_**i** = A_**i**j x_j |
| **Dummy index** | 두 번 나타남 → 합산되어 사라짐 | y_i = A_i**j** x_**j** |

---

### Exercise 풀이

#### (1) 행렬-벡터 곱 y = Ax를 Einstein Notation으로

**단계별 전개**:

```
y = Ax를 성분으로 쓰면:

y_1 = A_11 x_1 + A_12 x_2 + ... + A_1n x_n
y_2 = A_21 x_1 + A_22 x_2 + ... + A_2n x_n
...
y_i = A_i1 x_1 + A_i2 x_2 + ... + A_in x_n = Σ_j A_ij x_j
```

**Einstein Notation 답**:
$$y_i = A_{ij} x_j$$

**해석**:
- `i`는 free index → 결과 y의 인덱스
- `j`는 dummy index → 합산되어 사라짐
- "행렬 A의 i행을 벡터 x와 내적"

**NumPy 코드**:
```python
import numpy as np
y = np.einsum('ij,j->i', A, x)  # 명시적 Einstein
# 또는
y = A @ x  # 일반 행렬곱
```

---

#### (2) Frobenius 노름 ∥A∥²_F를 Einstein Notation으로

**Frobenius 노름이란?**

행렬의 "크기"를 재는 방법 중 하나:
- 모든 원소를 제곱해서 더함
- 벡터의 길이(L2 노름)를 행렬로 확장한 것

```
        [1 2]
A =     [3 4]

∥A∥²_F = 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
∥A∥_F = √30 ≈ 5.48
```

**일반 수식**:
$$\|A\|_F^2 = \sum_i \sum_j A_{ij}^2$$

**Einstein Notation 답**:
$$\|A\|_F^2 = A_{ij} A_{ij}$$

**해석**:
- `i`와 `j` 둘 다 두 번 나옴 → 둘 다 합산
- A의 각 원소를 자기 자신과 곱함 = 제곱
- 결과는 스칼라 (남는 인덱스 없음)

**NumPy 코드**:
```python
frobenius_sq = np.einsum('ij,ij->', A, A)
# 또는
frobenius_sq = np.sum(A**2)
# 또는
frobenius_sq = np.linalg.norm(A, 'fro')**2
```

---

### Question 1: C_ikl = A_ij B_jkl 의 차원

#### 단계별 분석

**주어진 정보**:
- A: 인덱스 i, j → 2차원 텐서 (행렬)
- B: 인덱스 j, k, l → 3차원 텐서
- C: 결과 텐서

**인덱스 분류**:
```
왼쪽 (C):  i, k, l  ← free indices (결과에 남음)
오른쪽:    i, j (from A)
           j, k, l (from B)

j가 두 번 나옴 → dummy index → 합산되어 사라짐!
```

**차원 계산**:
```
A: (I × J)
B: (J × K × L)

축약 후:
C: (I × K × L)  ← 3차원 텐서
```

**답: C는 3차원 텐서 (차원: I × K × L)**

#### 시각적 이해

```
A (2D):          B (3D):                 C (3D):
  j →              j →                     k →
i [█ █ █]      k [█ █ █]               i [█ █]
↓ [█ █ █]      ↓ [█ █ █]               ↓ [█ █]
               l [여러 층]              l [여러 층]

j축을 따라 합산 → j가 사라지고 (i, k, l)만 남음
```

#### 비유: 행렬 곱의 확장

행렬 곱 (2D × 2D → 2D):
```
(m × n) × (n × p) = (m × p)
   ↑         ↑
   n이 공통 → 사라짐
```

텐서 축약 (2D × 3D → 3D):
```
(I × J) × (J × K × L) = (I × K × L)
    ↑         ↑
    J가 공통 → 사라짐
```

---

### Question 2: 왜 메모리 연속성이 FLOPs보다 중요한가?

#### 배경: GPU의 작동 원리

**CPU vs GPU**:
```
CPU: 몇 개의 강력한 코어 → 복잡한 작업에 적합
GPU: 수천 개의 약한 코어 → 단순 작업 대량 병렬 처리
```

**GPU의 메모리 계층**:
```
속도 (빠름)           크기 (작음)
    ↑                    ↑
 레지스터 ────────────── 수 KB
 공유 메모리 ─────────── 수십 KB
 L1/L2 캐시 ──────────── 수 MB
 글로벌 메모리 (VRAM) ── 수십 GB
    ↓                    ↓
속도 (느림)           크기 (큼)
```

#### 핵심 문제: 메모리 병목

**FLOPs (연산량)**:
- GPU는 초당 수조(10^12) 개의 연산 가능
- 연산 자체는 매우 빠름

**메모리 대역폭**:
- 글로벌 메모리에서 데이터 읽기: ~1TB/s
- 하지만 랜덤 접근시 실제로는 훨씬 느림

```
시간 비교 (대략적):
- 곱셈 1회: ~0.001 ns
- 캐시 히트: ~1 ns
- 캐시 미스: ~100-300 ns  ← 곱셈보다 10만 배 느림!
```

#### 메모리 연속성 (Coalesced Access)

**연속 접근 (Coalesced)**:
```
메모리: [A₀][A₁][A₂][A₃][A₄][A₅][A₆][A₇]...

스레드 0 → A₀
스레드 1 → A₁
스레드 2 → A₂    한 번의 메모리 트랜잭션으로
스레드 3 → A₃    32개 스레드가 동시에 데이터 획득!
...
```

**비연속 접근 (Non-coalesced)**:
```
메모리: [A₀][B₀][C₀][D₀][A₁][B₁][C₁][D₁]...

스레드 0 → A₀  ← 별도 트랜잭션
스레드 1 → A₁  ← 별도 트랜잭션
스레드 2 → A₂  ← 별도 트랜잭션
...
32배 느려짐!
```

#### 실제 예시: 행 우선 vs 열 우선

**C/Python (Row-major)**:
```python
# 메모리에 저장된 순서: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], ...

# 빠름: 행 방향 접근 (연속)
for i in range(M):
    for j in range(N):
        access A[i, j]  # 연속된 메모리

# 느림: 열 방향 접근 (점프)
for j in range(N):
    for i in range(M):
        access A[i, j]  # 메모리에서 점프
```

#### 답변 요약

**FLOPs가 덜 중요한 이유**:
1. GPU 연산 속도는 이미 충분히 빠름
2. 병목은 "데이터를 가져오는 시간"

**메모리 연속성이 중요한 이유**:
1. **Coalesced access**: 연속 메모리 접근 시 대역폭 최대 활용
2. **캐시 효율**: 연속 데이터는 캐시에 함께 로드
3. **지연 숨기기**: GPU는 메모리 요청 중 다른 작업 가능, 단 연속 접근일 때만 효과적

**비유**:
```
도서관에서 책 10권 빌리기:

비연속 접근: 책장 A → 책장 Z → 책장 M → ... (매번 이동)
연속 접근:   책장 A에서 옆으로 쭉 10권 (한 번에)

읽는 속도(FLOPs)는 같지만, 가져오는 시간이 10배 차이!
```

---

## Challenge 2: 차원의 저주와 SVD/PCA

> **책 참조**: Section 1.2 (Matrix Decompositions), Exercise 1.2.2 (SVD for Matrix Completion)

### 배경 지식: SVD (Singular Value Decomposition)

#### SVD란 무엇인가?

**정의**: 임의의 m×n 행렬 A를 세 행렬의 곱으로 분해
$$A = U \Sigma V^T$$

**각 행렬의 의미**:
```
A (m×n): 원본 데이터
U (m×m): 왼쪽 특이벡터 (데이터 공간의 주요 방향)
Σ (m×n): 특이값 대각행렬 (각 방향의 중요도)
V (n×n): 오른쪽 특이벡터 (특징 공간의 주요 방향)
```

#### 기하학적 직관

**SVD = 회전 → 스케일링 → 회전**

```
원본 데이터 (단위원)
      ○

V^T 적용 (회전)
      ○

Σ 적용 (축 방향으로 늘리기/줄이기)
     ⬭  (타원)

U 적용 (다시 회전)
      ⬬  (회전된 타원)
```

**핵심**: 어떤 선형 변환이든 "회전-스케일-회전"으로 분해 가능!

#### 특이값의 의미

특이값 σ₁ ≥ σ₂ ≥ ... ≥ σᵣ (내림차순 정렬)

```
σ₁: 데이터가 가장 많이 퍼진 방향의 "퍼짐 정도"
σ₂: 두 번째로 많이 퍼진 방향
...
σᵣ: r번째 방향 (r = rank(A))
```

**에너지**: 특이값의 제곱 = 해당 방향의 "정보량"
$$\text{총 에너지} = \|A\|_F^2 = \sigma_1^2 + \sigma_2^2 + ... + \sigma_r^2$$

---

### Exercise 풀이: 90% 에너지 유지하는 k 찾기

#### 문제 설정

데이터 행렬 A가 주어졌을 때:
- SVD로 분해: A = UΣV^T
- 특이값: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ
- **질문**: 몇 개의 특이값(k)을 유지해야 90% 에너지를 보존하는가?

#### 수학적 정의

**총 에너지**:
$$E_{total} = \sum_{i=1}^{r} \sigma_i^2$$

**k개로 보존되는 에너지**:
$$E_k = \sum_{i=1}^{k} \sigma_i^2$$

**90% 보존 조건**:
$$\frac{E_k}{E_{total}} = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \geq 0.90$$

#### 구체적 예시

**데이터**: 100×50 행렬 A의 특이값
```
σ₁ = 50, σ₂ = 30, σ₃ = 20, σ₄ = 10, σ₅ = 5, 나머지 ≈ 0
```

**총 에너지**:
```
E_total = 50² + 30² + 20² + 10² + 5²
        = 2500 + 900 + 400 + 100 + 25
        = 3925
```

**누적 에너지**:
```
k=1: 2500/3925 = 63.7%  ❌
k=2: 3400/3925 = 86.6%  ❌
k=3: 3800/3925 = 96.8%  ✅  ← 최소 k = 3
```

**답: k = 3**

#### Python 코드

```python
import numpy as np

def find_k_for_energy(A, threshold=0.90):
    """90% 에너지를 보존하는 최소 k 찾기"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    total_energy = np.sum(s**2)
    cumulative_energy = np.cumsum(s**2) / total_energy

    k = np.searchsorted(cumulative_energy, threshold) + 1

    print(f"특이값: {s[:10]}")  # 상위 10개
    print(f"누적 에너지 비율: {cumulative_energy[:10]}")
    print(f"90% 에너지 보존을 위한 최소 k: {k}")

    return k

# 예시
A = np.random.randn(100, 50)
k = find_k_for_energy(A)
```

---

### Question 1: k 선택 = 메모리 vs 정보 손실 Trade-off

#### Low-Rank Approximation

**원본 저장**:
- A: m×n 행렬 → m·n 개의 숫자

**Rank-k 근사 저장**:
- U_k: m×k
- Σ_k: k×k (대각이므로 k개)
- V_k: k×n

**총 저장량**: m·k + k + k·n = k(m + n + 1) ≈ k(m + n)

#### 예시: MNIST 이미지

```
원본: 28×28 = 784 픽셀

SVD로 압축:
k=10:  28×10 + 10 + 10×28 = 570 (27% 압축) → 에너지 ~70%
k=50:  28×50 + 50 + 50×28 = 2850 (364% 원본) → 에너지 ~95%
k=100: 28×100 + 100 + 100×28 = 5700 → 에너지 ~99%
```

**잠깐!** k가 크면 오히려 원본보다 커질 수 있음!

**SVD 압축이 유리한 경우**:
$$k(m + n) < m \cdot n$$
$$k < \frac{m \cdot n}{m + n}$$

#### Trade-off 시각화

```
k (랭크)
│
│ ──────────────────────────── 정보 100% (원본)
│       ╱
│      ╱  정보 보존률
│     ╱
│    ╱
│───╱───────────────────────── 정보 90%
│  ╱
│ ╱
│╱
└──────────────────────────────→ 메모리 사용량

k↑: 정보↑, 메모리↑
k↓: 정보↓, 메모리↓
```

#### 실제 응용에서의 선택 기준

| 응용 | k 선택 기준 | 이유 |
|------|------------|------|
| 이미지 압축 | 90-95% 에너지 | 시각적 품질 유지 |
| 추천 시스템 | 수십~수백 | 사용자 선호 패턴 수 |
| 노이즈 제거 | 급격한 특이값 하락 지점 | 신호와 노이즈 분리 |
| LoRA (LLM) | 4-64 | 파라미터 효율 vs 성능 |

---

### Question 2: 왜 공분산 행렬 SVD가 A^T A 고유분해보다 나은가?

#### 문제 상황

**데이터**: N개 샘플, D개 특징
- N = 100,000 (샘플 수, 매우 큼)
- D = 100 (특징 수, 작음)
- 데이터 행렬 X: N × D

**목표**: 주성분(Principal Components) 찾기 = 데이터의 주요 방향

#### 방법 1: X 직접 SVD

```
X (N×D) = U (N×N) × Σ (N×D) × V^T (D×D)

문제: U가 N×N = 100,000×100,000 행렬!
     메모리: 80GB (float64 기준)
     계산: O(N²D) = O(10^12) 연산
```

#### 방법 2: X^T X 고유분해 (PCA)

**핵심 관계**:
$$X^T X = V \Sigma^T U^T \cdot U \Sigma V^T = V \Sigma^2 V^T$$

X^T X의 고유벡터 = X의 오른쪽 특이벡터 V!

```
X^T X (D×D) = 100×100 행렬

계산:
1. X^T X 계산: O(ND²) = O(10^9) 연산
2. D×D 고유분해: O(D³) = O(10^6) 연산
총: O(ND²) ≈ O(10^9)

메모리: 100×100 = 10,000 숫자 = 80KB
```

#### 비교 표

| 방법 | 행렬 크기 | 연산량 | 메모리 |
|------|-----------|--------|--------|
| X 직접 SVD | N×D → N×N | O(N²D) | O(N²) |
| X^T X 고유분해 | D×D | O(ND² + D³) | O(D²) |

**N >> D일 때**: 방법 2가 압도적으로 유리!

```
예시 (N=100,000, D=100):
방법 1: 10^12 연산, 80GB 메모리
방법 2: 10^9 연산, 80KB 메모리
→ 1000배 빠르고, 100만 배 적은 메모리!
```

#### 직관적 설명

**X 직접 SVD**:
- 모든 N개 샘플 각각에 대한 정보(U)를 계산
- 하지만 PCA에서는 V(주성분)만 필요!

**X^T X 고유분해**:
- 먼저 공분산 구조만 요약 (D×D로 압축)
- 이 작은 행렬에서 주성분 추출
- 불필요한 정보(각 샘플의 좌표)는 계산하지 않음

**비유**:
```
문제: 100만 명의 키, 몸무게로 체형 패턴 찾기

방법 1: 100만 명 각각의 정보 모두 분석
방법 2: "키-몸무게 상관관계"만 계산 (2×2 행렬)
        → 이 작은 표에서 패턴 추출

당연히 방법 2가 효율적!
```

#### Python 코드 비교

```python
import numpy as np
import time

N, D = 100000, 100
X = np.random.randn(N, D)

# 방법 1: 직접 SVD (truncated)
start = time.time()
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print(f"직접 SVD: {time.time() - start:.2f}초")

# 방법 2: 공분산 행렬 고유분해
start = time.time()
cov = X.T @ X / N  # D×D 공분산 행렬
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print(f"공분산 고유분해: {time.time() - start:.2f}초")

# 결과 비교: V와 eigenvectors는 (부호 제외) 동일해야 함
print(f"V와 eigenvectors 일치: {np.allclose(np.abs(Vt.T), np.abs(eigenvectors))}")
```

---

## Challenge 3: Chain Rule과 역전파 (Backpropagation)

> **책 참조**: Chapter 2 (Calculus), Section 2.1 (Automatic Differentiation), Exercise 2.1.1

### 배경 지식: 미분과 Chain Rule

#### 미분의 기본

**미분이란?**: 함수의 "기울기" 또는 "변화율"

```
f(x) = x²

x=2에서 미분값:
f'(2) = 2×2 = 4

의미: x가 2 근처에서 조금 변할 때,
      f(x)는 약 4배 빠르게 변한다
```

#### Chain Rule (연쇄 법칙)

**문제**: 합성함수의 미분

```
h(x) = f(g(x))
예: h(x) = sin(x²)  where f(u) = sin(u), g(x) = x²

h'(x) = ?
```

**Chain Rule**:
$$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

```
h'(x) = f'(g(x)) × g'(x)
      = cos(x²) × 2x
```

**직관**: "연쇄적인 변화율의 곱"
```
x가 변하면 → g가 변하고 → f가 변한다
x → g → f
   ×     ×
  dg/dx  df/dg
```

#### 다변수로 확장: Jacobian

**단일 변수**: 미분값 = 스칼라
**다변수 (벡터 → 벡터)**: Jacobian = 행렬

```
f: R^n → R^m
y = f(x)  where x ∈ R^n, y ∈ R^m

Jacobian J ∈ R^(m×n):
J_ij = ∂y_i / ∂x_j
```

**의미**: J[i,j] = "x_j가 변할 때 y_i가 얼마나 변하는가"

---

### Exercise (a): 2층 신경망의 Jacobian

#### 네트워크 구조

$$f(x) = W_2 \sigma(W_1 x)$$

**변수들**:
- 입력: x ∈ R^d
- 첫 번째 가중치: W₁ ∈ R^(h×d)
- 활성화 함수: σ (element-wise, 예: ReLU, sigmoid)
- 두 번째 가중치: W₂ ∈ R^(m×h)
- 출력: f(x) ∈ R^m

#### 순전파 단계별 분해

```
x ∈ R^d
    │
    ↓ z₁ = W₁x
z₁ ∈ R^h
    │
    ↓ a₁ = σ(z₁)
a₁ ∈ R^h
    │
    ↓ f = W₂a₁
f ∈ R^m
```

#### 각 단계의 Jacobian

**Step 1: z₁ = W₁x**
$$\frac{\partial z_1}{\partial x} = W_1 \in \mathbb{R}^{h \times d}$$

설명: 선형 변환의 Jacobian = 변환 행렬 자체

**Step 2: a₁ = σ(z₁)**
$$\frac{\partial a_1}{\partial z_1} = \text{diag}(\sigma'(z_1)) \in \mathbb{R}^{h \times h}$$

설명:
- σ가 element-wise이므로 각 원소 독립
- 대각행렬 형태
- σ'(z₁) = [σ'(z₁,₁), σ'(z₁,₂), ..., σ'(z₁,h)]

**Step 3: f = W₂a₁**
$$\frac{\partial f}{\partial a_1} = W_2 \in \mathbb{R}^{m \times h}$$

#### 전체 Jacobian (Chain Rule 적용)

$$J = \frac{\partial f}{\partial x} = \frac{\partial f}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial x}$$

$$\boxed{J = W_2 \cdot \text{diag}(\sigma'(z_1)) \cdot W_1}$$

**차원 확인**:
```
(m×h) · (h×h) · (h×d) = (m×d) ✓
```

---

### Exercise (b): ∂L/∂W₁ 유도

#### 손실 함수 설정

- L: 스칼라 손실 함수 (예: MSE, Cross-entropy)
- 목표: L을 W₁에 대해 미분

#### Chain Rule 경로

```
W₁ → z₁ → a₁ → f → L
```

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

#### 각 항 계산

**∂L/∂f**: 주어진 값 (손실의 출력에 대한 기울기)
- 크기: R^m (또는 R^(1×m) row vector)

**∂f/∂a₁ = W₂**:
- 크기: R^(m×h)

**∂a₁/∂z₁ = diag(σ'(z₁))**:
- 크기: R^(h×h)

**∂z₁/∂W₁**:
- z₁ = W₁x에서 W₁에 대한 미분
- 이건 텐서가 됨 (R^h → R^(h×d))
- 결과적으로 ∂z₁,ᵢ/∂W₁,ᵢⱼ = xⱼ

#### 최종 결과 유도

**Step 1**: 역전파된 오차 계산
$$\delta_1 = \left( W_2^T \frac{\partial L}{\partial f} \right) \odot \sigma'(z_1)$$

- W₂^T (∂L/∂f): 오차를 이전 층으로 전파 (m→h)
- ⊙ σ'(z₁): 활성화 함수의 기울기로 스케일

**Step 2**: 가중치 기울기
$$\boxed{\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T}$$

**차원 확인**:
```
δ₁: (h×1)
x^T: (1×d)
결과: (h×d) = W₁의 크기 ✓
```

#### 직관적 이해

```
∂L/∂W₁ = δ₁ · x^T

δ₁: "이 뉴런이 얼마나 책임이 있는가" (역전파된 오차)
x^T: "어떤 입력이 들어왔는가"

결과: "책임 × 입력 = 가중치 업데이트량"
```

---

### Question 1: Forward Pass와 Backward Pass의 분리

#### 데이터 흐름 관점

**Forward Pass (순방향)**:
```
목적: 함수값 f(x) 계산
방향: 입력 → 출력

x → [W₁] → z₁ → [σ] → a₁ → [W₂] → f → [Loss] → L
 →     →     →     →     →     →     →
```

**Backward Pass (역방향)**:
```
목적: 기울기 ∇L 계산
방향: 출력 → 입력

∂L/∂W₁ ← δ₁ ← [W₂^T] ← ⊙σ' ← ∂L/∂f ← 1
        ←      ←      ←     ←
```

#### 왜 분리되어야 하는가?

**1. 인과관계 (Causality)**

Forward: f(x)를 계산하려면 x부터 시작해야 함
- z₁을 모르면 a₁을 계산할 수 없음
- a₁을 모르면 f를 계산할 수 없음

Backward: ∂L/∂x를 계산하려면 L부터 시작해야 함
- ∂L/∂f를 모르면 ∂L/∂a₁을 계산할 수 없음
- Chain Rule은 "끝에서부터" 적용해야 함

**2. 중간값 저장 필요**

```python
# Forward
z1 = W1 @ x        # z1 저장 필요!
a1 = sigmoid(z1)   # a1, z1 저장 필요!
f = W2 @ a1

# Backward (저장된 값 사용)
dL_da1 = W2.T @ dL_df
dL_dz1 = dL_da1 * sigmoid_derivative(z1)  # z1 필요!
dL_dW1 = dL_dz1 @ x.T                      # x 필요!
```

**3. 효율성**

Forward와 Backward를 동시에 할 수 없음:
- Forward 중에는 아직 L을 모름
- Backward는 Forward 결과가 있어야 시작 가능

#### 비유: 산 등반

```
Forward (등반):
산을 오르면서 "여기 왼쪽으로 갔다", "저기서 오른쪽으로 갔다" 기록

Backward (하산):
정상에서 시작해서, 기록을 거꾸로 보며
"각 갈림길에서 경사가 어땠는지" 계산
```

---

### Question 2: Exploding Gradient와 Gradient Clipping

#### Exploding Gradient 문제

**깊은 네트워크에서의 기울기**:

T층 네트워크의 경우:
$$\frac{\partial L}{\partial x} = W_T \cdot D_{T-1} \cdot W_{T-1} \cdot D_{T-2} \cdot ... \cdot W_1$$

여기서 D_i = diag(σ'(z_i))

**문제 상황**:

각 가중치 행렬의 spectral norm (최대 특이값)이 1보다 크면:
$$\|W_i\| > 1 \quad \Rightarrow \quad \left\|\prod_{i=1}^{T} W_i\right\| \sim \|W\|^T$$

```
예: ∥W∥ = 1.1, T = 100
    ∥기울기∥ ∼ 1.1^100 ≈ 13,781

    → 기울기가 폭발!
```

**결과**:
- 가중치 업데이트가 너무 큼
- 학습이 불안정하거나 발산
- NaN 값 발생

#### Gradient Clipping

**아이디어**: 기울기 크기가 임계값을 넘으면 스케일 다운

**수식**:
$$g_{clipped} = \begin{cases} g & \text{if } \|g\| \leq \tau \\ \tau \cdot \frac{g}{\|g\|} & \text{if } \|g\| > \tau \end{cases}$$

**코드**:
```python
def clip_gradient(g, max_norm):
    """기울기 클리핑"""
    norm = np.linalg.norm(g)
    if norm > max_norm:
        g = g * (max_norm / norm)
    return g
```

**PyTorch에서**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 기하학적 해석

```
클리핑 전:                    클리핑 후:
      ↑                           ↑
      │ (기울기 = 100)            │ (기울기 = 1, 같은 방향)
      │                           ▲
      │                          ╱
      ○ 현재 위치               ○ 현재 위치

"방향은 유지하되, 보폭만 제한"
```

#### 클리핑의 효과

**장점**:
1. 학습 안정화: 급격한 파라미터 변화 방지
2. NaN 방지: 수치적 오버플로우 예방
3. 방향 보존: 기울기가 가리키는 방향은 유지

**단점**:
1. 학습 속도 저하: 큰 기울기가 필요한 경우에도 제한
2. 하이퍼파라미터 추가: 적절한 τ 선택 필요

#### 다른 해결책들

| 방법 | 설명 |
|------|------|
| **Gradient Clipping** | 기울기 크기 직접 제한 |
| **Weight Initialization** | Xavier/He 초기화로 특이값 ≈ 1 유지 |
| **Batch Normalization** | 각 층의 활성화 분포 정규화 |
| **Residual Connections** | 기울기가 shortcut으로 직접 전달 |
| **LSTM/GRU** | RNN에서 gate로 기울기 흐름 제어 |

---

## Challenge 4: ODE 기반 파라미터 추정

> **책 참조**: Chapter 2, Section 2.2 (Differential Equations), Exercise 2.2.3 (Regression and Parameter Estimation for an ODE)

### 배경 지식: 미분방정식 (ODE)

#### ODE란?

**정의**: 미지 함수와 그 도함수 사이의 관계식

```
dx/dt = f(t, x)

x(t): 시간에 따라 변하는 양
dx/dt: x의 변화율
f: x가 어떻게 변하는지 규정하는 함수
```

#### 예시: 지수 감쇠

$$\frac{dx}{dt} = -\gamma x$$

**해석**: "x의 변화율은 현재 x 값에 비례하며 감소한다"

**해**: x(t) = x₀ e^(-γt)

```
시간 →
x ──────────────────
   \
    \
     \
      \_________     γ가 크면: 빠르게 감소
                     γ가 작으면: 천천히 감소
```

#### 파라미터 추정 문제

**주어진 것**: 노이즈가 섞인 관측 데이터 {(t_i, x_i)}
**목표**: 모델 파라미터 γ 추정

---

### Exercise: ẋ = -γx + cos(2t) 에서 γ 추정

#### 모델 설정

$$\dot{x} = -\gamma x + \cos(2t)$$

**해석**:
- -γx: 감쇠 항 (x가 0으로 돌아가려는 힘)
- cos(2t): 외부 강제력 (주기적으로 밀어주는 힘)

#### 방법 1: 직접 회귀 (x(t) 피팅)

**절차**:
1. ODE를 해석적으로 풀어서 x(t) 형태 얻기
2. x(t) = f(t; γ)를 데이터에 피팅
3. 최소제곱법으로 γ 추정

**문제점**:
- ODE 해가 복잡할 수 있음
- 초기조건에 민감
- 노이즈가 누적됨

#### 방법 2: ODE 기반 회귀 (ẋ 피팅)

**절차**:

**Step 1**: 데이터에서 미분값 추정
```
ẋᵢ ≈ (xᵢ₊₁ - xᵢ₋₁) / (2Δt)  (중앙 차분)
```

**Step 2**: 선형 회귀 문제로 변환

$$\dot{x}_i = -\gamma x_i + \cos(2t_i)$$

정리하면:
$$\dot{x}_i = [-x_i, \cos(2t_i)] \cdot [\gamma, 1]^T$$

**Step 3**: 행렬 형태로 작성

$$\underbrace{\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \\ \vdots \\ \dot{x}_N \end{bmatrix}}_{\mathbf{b}} = \underbrace{\begin{bmatrix} -x_1 & \cos(2t_1) \\ -x_2 & \cos(2t_2) \\ \vdots & \vdots \\ -x_N & \cos(2t_N) \end{bmatrix}}_{\mathbf{A}} \underbrace{\begin{bmatrix} \gamma \\ 1 \end{bmatrix}}_{\boldsymbol{\theta}}$$

**Step 4**: 최소제곱법
$$\hat{\boldsymbol{\theta}} = (A^T A)^{-1} A^T \mathbf{b}$$

#### Python 코드

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 실제 파라미터
gamma_true = 0.5

# ODE 정의
def ode_rhs(x, t, gamma):
    return -gamma * x + np.cos(2*t)

# 데이터 생성
t = np.linspace(0, 10, 100)
x0 = 1.0
x_true = odeint(ode_rhs, x0, t, args=(gamma_true,)).flatten()

# 노이즈 추가
noise_level = 0.1
x_noisy = x_true + noise_level * np.random.randn(len(t))

# ODE 기반 회귀
# Step 1: 미분값 추정 (중앙 차분)
dt = t[1] - t[0]
x_dot = np.zeros(len(t) - 2)
for i in range(1, len(t) - 1):
    x_dot[i-1] = (x_noisy[i+1] - x_noisy[i-1]) / (2 * dt)

# Step 2: 회귀 행렬 구성
t_inner = t[1:-1]
x_inner = x_noisy[1:-1]

A = np.column_stack([-x_inner, np.cos(2*t_inner)])
b = x_dot

# Step 3: 최소제곱법
theta, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
gamma_estimated = theta[0]

print(f"실제 γ: {gamma_true}")
print(f"추정 γ: {gamma_estimated:.4f}")
print(f"오차: {abs(gamma_estimated - gamma_true):.4f}")
```

---

### Question 1: ODE 회귀 vs 직접 회귀의 장점

#### 노이즈의 영향 비교

**직접 회귀 (x(t) 피팅)**:

```
문제: 적분의 성질

x(t) = x₀ + ∫₀ᵗ ẋ(s) ds

노이즈가 적분을 통해 누적됨!

예: 매 시점마다 ε만큼 오차
    t=10까지 100 스텝 → 총 오차 ~100ε
```

**ODE 회귀 (ẋ 피팅)**:

```
장점: 미분은 "현재 시점" 정보만 사용

ẋᵢ ≈ (xᵢ₊₁ - xᵢ₋₁) / (2Δt)

각 ẋᵢ의 오차는 독립적!
회귀에서 평균화됨 → 총 오차 ~ε/√N
```

#### 시각적 비교

```
직접 회귀:
x ─────────────
  ~~~~~~~~~     ← 궤적 전체를 맞춰야 함
   ~~~~~~~~~        누적 오차 영향 큼
    ~~~~~~~

ODE 회귀:
각 점에서 기울기(접선)만 맞추면 됨

    ╱ ← 이 기울기가 맞는가?
   ○
      ╱ ← 이 기울기가 맞는가?
     ○
```

#### 핵심 차이

| 측면 | 직접 회귀 | ODE 회귀 |
|------|-----------|----------|
| 피팅 대상 | 전체 궤적 x(t) | 국소 기울기 ẋ(t) |
| 노이즈 영향 | 누적 (적분) | 독립 (평균화) |
| 계산 복잡도 | ODE 풀이 필요 | 선형 회귀만 |
| 초기조건 민감도 | 높음 | 낮음 |

---

### Question 2: ODE가 추정을 안정화시키는 핵심 성질

#### 물리적 제약조건 (Physics Constraint)

**핵심**: ODE는 "물리적으로 가능한 해"만 허용

```
ODE: ẋ = -γx + cos(2t)

이것이 암묵적으로 말하는 것:
1. x(t)는 연속적이고 미분 가능하다
2. 변화율은 현재 상태에만 의존한다 (마르코프 성질)
3. 감쇠와 외력의 형태가 정해져 있다
```

#### 자유도 제한

**직접 회귀 (다항식)**:
```
x(t) ≈ a₀ + a₁t + a₂t² + ... + aₙtⁿ

자유도: n+1 (계수의 개수)
많은 자유도 → 노이즈까지 피팅할 위험 (과적합)
```

**ODE 회귀**:
```
ẋ = -γx + cos(2t)

자유도: 1 (γ 하나)
적은 자유도 → 물리적으로 의미 있는 해만 가능
```

#### 정규화 효과 (Regularization)

ODE 구조 자체가 정규화 역할:
- "이 데이터는 감쇠 진동에서 왔다"는 가정
- 노이즈로 인한 비물리적 패턴은 자동 무시

```
노이즈 패턴:    ~~~~↗~~~~↘~~~~
물리적 패턴:   ─────↘─────↘───

ODE 회귀는 물리적 패턴만 찾음
```

#### 비유: 범인 찾기

**직접 회귀**: "이 사람들 중 범인이 누구인가?" (자유도 높음)
- 무고한 사람도 의심할 수 있음

**ODE 회귀**: "범인은 왼손잡이다"라는 물리적 증거가 있음 (제약조건)
- 오른손잡이는 자동으로 제외
- 용의자 범위가 크게 좁아짐

#### 안정성의 수학적 기원

**Well-posedness**: ODE의 해가 존재하고 유일하며 연속적으로 의존

```
데이터가 조금 변해도 → 추정된 γ는 조금만 변함
(Lipschitz 연속성)
```

**Ill-posedness**: 직접 회귀에서는 작은 노이즈가 큰 변화 유발 가능

```
적분 연산자의 역문제는 종종 ill-posed
→ 노이즈 증폭
```

---

# Lecture 2 Challenges (Ch. 3-4: Optimization & Deep Learning)

---

## Challenge 5: Optimization Landscapes (최적화 지형)

> **책 참조**: Chapter 3 (Optimization), Section 3.2 (Convex Optimization), Exercise 3.2.1 (GD Trajectories)

### 배경 지식: Gradient Descent

#### 경사하강법이란?

**목표**: 손실 함수 L(θ)를 최소화하는 파라미터 θ 찾기

**알고리즘**:
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

- η: 학습률 (learning rate)
- ∇L: 손실 함수의 기울기

**직관**: "공이 언덕을 따라 굴러 내려가는 것"

```
손실 함수 (2D 예시):

    높음
    ↑   ╱╲
    │  ╱  ╲
    │ ╱    ╲    ← 경사를 따라 내려감
    │╱      ╲
    └────────→ θ
              최소점
```

---

### Convex vs Non-Convex 함수

#### Convex (볼록) 함수

**정의**: 임의의 두 점을 잇는 선분이 함수 그래프 위에 있음

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**시각적**:
```
      ∪
     / \
    /   \
   /     \
  /       \
 ←─────────→
   유일한 최소점
```

**특징**:
- 전역 최소점 = 지역 최소점 (하나뿐!)
- 기울기 = 0인 점은 최소점
- 어디서 시작해도 같은 곳으로 수렴

**예시**: L(θ) = θ², 선형 회귀의 MSE 손실

#### Non-Convex (비볼록) 함수

**시각적**:
```
      ∿∿∿
     /    \      /\
    /      \    /  \
   /        \  /    \
  /          \/      \
 ←───────────────────→
 지역 최소  전역 최소  지역 최소
```

**특징**:
- 여러 개의 지역 최소점
- 시작점에 따라 다른 최소점으로 수렴
- 신경망의 손실 함수는 대부분 non-convex

---

### Question 1: Convex에서 모든 궤적이 같은 최소점으로 수렴하는 이유

#### 수학적 이유

**Convex 함수의 핵심 성질**:

1. **유일성**: 전역 최소점이 유일하게 존재
2. **방향성**: 기울기는 항상 최소점 "쪽"을 가리킴

**수식**:
Convex 함수 f에서, 최소점 θ*에 대해:
$$\nabla f(\theta) \cdot (\theta - \theta^*) \geq 0$$

**해석**: 기울기의 반대 방향으로 가면 항상 θ*에 가까워짐

#### 기하학적 직관

```
어디서 시작해도:

    시작점 A          시작점 B
         \              /
          \            /
           \          /
            \        /
             ↘      ↙
              유일한 최소점

모든 길이 로마로 통한다!
```

**왜 non-convex에서는 안 되는가?**
```
         시작 A            시작 B
            ↓                 ↓
    ╱╲     │       ╱╲       │
   ╱  ╲    ↓      ╱  ╲      ↓
  ╱    ↓──→최소1  ╱    ╲──→최소2
 ╱                      ╲

각자 가장 가까운 최소점으로 빠짐
```

#### Learning Rate의 영향

**η가 너무 작으면**:
```
→ → → → → → → 최소점
수렴은 하지만 매우 느림
```

**η가 적당하면**:
```
─ ─ ─ → 최소점
효율적으로 수렴
```

**η가 너무 크면**:
```
←→←→←→ 최소점 주변에서 진동
또는
→→→→→→→→→ 발산 (튕겨나감)
```

**수학적 조건**:
Convex quadratic f(θ) = ½θᵀHθ에서
수렴 조건: η < 2/λ_max(H)

```
λ_max = 가장 가파른 방향의 곡률
η가 이보다 크면 그 방향에서 발산
```

---

### Question 2: Non-Convex에서 초기화와 노이즈의 역할

#### Basin of Attraction (흡인 영역)

**정의**: 특정 최소점으로 수렴하는 초기점들의 집합

```
손실 지형:
      ∿      ∿
     / \    / \
    /   \  /   \
   /     \/     \
  /   최소1   최소2  \

Basin 1:  ████████░░░░░░░░
Basin 2:  ░░░░░░░░████████

시작점이 어느 basin에 있느냐에 따라 결과 결정
```

**예시 코드**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Non-convex 함수: 두 개의 최소점
def f(x):
    return x**4 - 4*x**2 + x

def grad_f(x):
    return 4*x**3 - 8*x + 1

# 여러 초기점에서 시작
initial_points = np.linspace(-2.5, 2.5, 20)
eta = 0.01

for x0 in initial_points:
    x = x0
    trajectory = [x]
    for _ in range(1000):
        x = x - eta * grad_f(x)
        trajectory.append(x)

    final = trajectory[-1]
    # 어느 최소점으로 수렴했는가?
    if final < 0:
        color = 'blue'  # 왼쪽 최소점
    else:
        color = 'red'   # 오른쪽 최소점
    plt.plot(trajectory[:100], alpha=0.5, color=color)

plt.title("Different initializations → Different minima")
plt.show()
```

#### SGD 노이즈의 역할

**SGD (Stochastic Gradient Descent)**:
$$\theta_{t+1} = \theta_t - \eta \nabla L_{batch}(\theta_t)$$

미니배치의 기울기 ≠ 전체 기울기 → 노이즈 발생!

**노이즈의 긍정적 효과**:

**1. Saddle Point 탈출**

```
Saddle Point: 한 방향으로는 최소, 다른 방향으로는 최대

      ↑ (이 방향은 최대)
      │
  ────┼────→ (이 방향은 최소)
      │
      saddle

순수 GD: 정확히 saddle에서 시작하면 탈출 불가
SGD: 노이즈가 옆으로 밀어줌 → 탈출!
```

**2. 얕은 최소점 탈출**

```
손실 지형:
    ╱╲
   ╱  ╲  ╱────╲
  ╱    ╲╱      ╲
 얕은    깊은
 최소    최소

순수 GD: 얕은 최소점에 빠지면 탈출 불가
SGD: 노이즈로 튕겨나와서 더 좋은 곳 탐색
```

**3. Flat Minima 선호**

```
Sharp minimum:        Flat minimum:
     ╱╲                 ╱──────╲
    ╱  ╲               ╱        ╲
   ╱    ╲             ╱          ╲
  ╱      ╲           ╱            ╲

Sharp: 노이즈에 불안정 → SGD가 빠져나감
Flat: 노이즈에도 안정 → SGD가 머무름

→ Flat minima는 일반화 성능이 좋은 경향!
```

#### 비유: 공 굴리기

```
순수 GD = 완벽히 매끄러운 공
→ 어떤 웅덩이든 빠지면 탈출 불가

SGD = 떨리는 공 (진동하면서 굴러감)
→ 얕은 웅덩이: 떨림으로 탈출
→ 깊은 웅덩이: 떨려도 빠져나오지 못함 (안정적으로 정착)
```

---

## Challenge 6: Tiny Transformer와 Optimizer

> **책 참조**: Chapter 3, Section 3.4 (Adaptive Learning Rate Methods), Exercise 3.5.1 (Tiny Transformers)

### 배경 지식: Optimizer 종류

#### SGD (Stochastic Gradient Descent)

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

- g_t: 미니배치 기울기
- η: 학습률 (상수)

**특징**: 단순하지만 학습률 선택이 어려움

#### Momentum

$$v_t = \beta v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

**직관**: "관성을 가진 공"
- 이전 방향의 영향을 받음
- 진동 감소, 평탄한 곳에서도 속도 유지

#### RMSProp

$$v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{g_t}{\sqrt{v_t + \epsilon}}$$

**직관**: "각 파라미터별 적응적 학습률"
- 기울기가 컸던 파라미터: 작은 학습률
- 기울기가 작았던 파라미터: 큰 학습률

#### Adam (Adaptive Moment Estimation)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1차 모멘트: 방향)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2차 모멘트: 크기)}$$

**Bias Correction**:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**업데이트**:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**직관**: RMSProp + Momentum + Bias Correction

---

### Question 1: 시퀀스 구조가 Optimizer 성능에 미치는 영향

#### 데이터 유형별 특성

**반복 패턴 (예: "abcabcabc")**:
```
특징:
- 예측이 쉬움 (다음 글자가 뻔함)
- 기울기가 작고 안정적
- Loss가 빠르게 감소

결과:
- SGD도 잘 작동
- Adam이 더 빠르지만 차이 작음
```

**구두점/공백이 많은 텍스트**:
```
특징:
- 구두점은 드물게 등장 → 희소(sparse)
- 등장할 때 큰 기울기 발생
- 기울기 분산이 큼

결과:
- SGD: 희소 업데이트에 비효율
- Adam: 적응적 학습률로 잘 처리
```

**자연어 텍스트**:
```
특징:
- 다양한 패턴 혼재
- 장거리 의존성 (문장 초반이 끝에 영향)
- 기울기 방향/크기 모두 불안정

결과:
- SGD: 수렴 느리고 불안정
- Adam: 안정적 수렴
- RMSProp: 보통 수준
```

#### 수렴 속도 비교 (전형적)

```
Loss
│                         반복 패턴
│ ╲
│  ╲          SGD
│   ╲_________
│    ╲         자연어 (SGD)
│     ╲╲
│      ╲ ╲
│       ╲  ╲_____ 자연어 (Adam)
│        ╲
│         ╲______ 반복 패턴 (Adam)
└─────────────────→ 에폭
```

---

### Question 2: RMSProp 정체 vs Adam

#### RMSProp의 문제

**시나리오**: 초기에 큰 기울기 발생

```
t=1: g₁ = 100 (가파른 절벽)
     v₁ = 0.9×0 + 0.1×100² = 1000
     업데이트 = 100/√1000 ≈ 3.16

t=2: g₂ = 1 (평탄한 지역)
     v₂ = 0.9×1000 + 0.1×1² = 900.1
     업데이트 = 1/√900.1 ≈ 0.033

문제: v₂가 여전히 크기 때문에
      업데이트가 너무 작아짐!
```

**기하학적 해석**:

```
손실 지형:
    ↓ 시작점
    │
    │ 절벽 (큰 기울기)
    │
    └────────── 평탄한 지역 (작은 기울기)

RMSProp: 절벽에서 v가 커짐
         → 평탄 지역에서도 v가 계속 큼
         → 거북이처럼 느리게 진행
```

#### Adam의 해결책

**1. Bias Correction**

```python
# RMSProp (bias correction 없음)
v_t = beta * v_{t-1} + (1-beta) * g_t**2

# Adam
v_t = beta * v_{t-1} + (1-beta) * g_t**2
v_corrected = v_t / (1 - beta**t)  # 초기 bias 보정
```

**효과**:
```
t=1일 때 RMSProp:  v₁ = 0.1 × g₁² (너무 작거나 큼)
t=1일 때 Adam:     v₁ / (1-0.999) = v₁ / 0.001 (적절히 스케일)
```

**2. Momentum (1차 모멘트)**

```
RMSProp: 현재 기울기 g_t만 사용
Adam:    m_t = β₁ m_{t-1} + (1-β₁) g_t

m_t는 기울기의 이동평균 → "어느 방향으로 꾸준히 가야 하는가"
```

**기하학적 비교**:

```
RMSProp:
시간 →  g: [100, 1, 1, 1, 1, ...]
        v: [1000, 900, 810, 729, ...]  ← 천천히 감소
        업데이트: 작음

Adam:
시간 →  g: [100, 1, 1, 1, 1, ...]
        m: [10, 1.9, 1.1, 1.0, ...]   ← 방향 유지
        v: [1000, 900, ...]           ← 비슷
        하지만 bias correction으로 초기 문제 완화
        m으로 방향 정보 유지 → 꾸준히 전진
```

#### 요약 비교

| 측면 | RMSProp | Adam |
|------|---------|------|
| 학습률 적응 | ✅ | ✅ |
| Momentum | ❌ | ✅ |
| Bias Correction | ❌ | ✅ |
| 초기 큰 기울기 후 | 정체 가능 | 잘 회복 |
| 일반적 성능 | 좋음 | 더 좋음 |

---

### Question 3: Confusion Matrix가 보여주는 것

#### Confusion Matrix란?

```
              예측
             A  B  C
        A   90  5  5
실제    B    3 92  5
        C    2  3 95

대각선: 정확히 분류된 샘플 수
비대각선: 혼동된 샘플 (실제 i인데 j로 예측)
```

#### 분석 방법

**1. 클래스별 정확도**:
```
A: 90/(90+5+5) = 90%
B: 92/(3+92+5) = 92%
C: 95/(2+3+95) = 95%
```

**2. 혼동 패턴**:
```
A↔B 혼동: 5+3 = 8번
B↔C 혼동: 5+3 = 8번
A↔C 혼동: 5+2 = 7번

→ 모든 클래스 쌍이 비슷하게 혼동됨
   또는 특정 쌍이 더 많이 혼동되면 그 클래스들이 유사
```

#### Optimization + Data Geometry의 관계

**1. 데이터 기하학 반영**:

```
Feature space에서의 클래스 분포:

A●●●
  ●●●●
    B●●●●
        ●●●●
          C●●●

A와 B가 겹침 → Confusion Matrix에서 A↔B 혼동 많음
B와 C가 겹침 → B↔C 혼동 많음
A와 C는 멀음 → A↔C 혼동 적음
```

**2. Optimizer의 영향**:

```
SGD:
- 모든 클래스에 비슷한 노력
- 전반적으로 균일한 성능

Adam:
- 어려운 샘플에 더 적응
- 기울기가 큰(틀리기 쉬운) 샘플에 집중
- 결과: 어려운 클래스 경계가 더 잘 학습됨
```

**3. Confusion Matrix로 진단**:

```
좋은 학습:
    A   B   C
A  95   3   2
B   2  96   2
C   1   2  97
→ 대각선 집중 = 클래스 잘 분리됨

나쁜 학습:
    A   B   C
A  40  30  30
B  25  50  25
C  20  20  60
→ 분산됨 = 클래스 혼동 심함
```

**4. 개선 방향 제시**:

```
B↔C 혼동이 특히 많다면:
→ B와 C를 구분하는 특징이 부족
→ 더 많은 B, C 샘플 필요
→ 또는 B, C 구분에 특화된 특징 추가
```

---

## Challenge 7: CNN의 Capacity와 Inductive Bias

> **책 참조**: Chapter 4 (Neural Networks), Section 4.1.3 (Simple CNN), Exercise 4.1.3 (Filter Count and Size)

### 배경 지식: CNN 구조

#### Convolutional Layer

```
입력 (28×28×1)
    │
    ↓ 3×3 Conv, 32 filters
Feature Map (26×26×32)
    │
    ↓ 3×3 Conv, 64 filters
Feature Map (24×24×64)
    │
    ↓ Pooling
(12×12×64)
    │
    ↓ Flatten + FC
출력 (10 classes)
```

#### 핵심 개념

**Filter (Kernel)**: 특징을 검출하는 작은 가중치 행렬
```
예: 3×3 Edge detector
[-1, 0, 1]
[-1, 0, 1]
[-1, 0, 1]
```

**Channel**: 같은 위치에서 검출하는 특징의 종류
```
32 channels = 32가지 다른 특징을 검출
(가로선, 세로선, 대각선, 곡선, ...)
```

**Receptive Field**: 출력 1픽셀이 "보는" 입력 영역
```
3×3 conv 1층: 3×3 receptive field
3×3 conv 2층: 5×5 receptive field
3×3 conv 3층: 7×7 receptive field
```

---

### Question 1: Channel 수 증가의 영향

#### Expressiveness (표현력)

```
채널 수 ↑ → 검출 가능한 특징 종류 ↑

4 채널:
- 가로선, 세로선, 대각선 2종
- 기본적인 특징만

32 채널:
- 위 4개 + 곡선, 모서리, 교차점,
  다양한 각도의 선, 두께 변화...
- 풍부한 특징

128 채널:
- 위 32개 + 매우 세밀한 패턴
- 특정 숫자의 고유한 스타일까지
```

#### Overfitting 위험

**파라미터 수 계산**:
```
Conv layer: kernel_size² × in_channels × out_channels

예: 3×3 kernel, 1→32 channels
    9 × 1 × 32 = 288 파라미터

예: 3×3 kernel, 32→128 channels
    9 × 32 × 128 = 36,864 파라미터
```

**MNIST에서의 관찰**:

```
채널 수   파라미터    Train Acc   Test Acc   Gap
4         ~1K        95%         94%        1%
16        ~10K       98%         97%        1%
64        ~100K      99.5%       98%        1.5%
256       ~1M        99.9%       97%        2.9%  ← 과적합!
```

**왜 MNIST에서 많은 채널이 불필요한가?**

```
MNIST 특징:
- 10개 클래스 (0-9)
- 간단한 획 패턴
- 배경 단순 (흑백)

필요한 특징:
- 곡선 (0, 6, 8, 9)
- 직선 (1, 4, 7)
- 교차점 (4, 8)
→ 수십 개면 충분!

채널 256개:
- 불필요하게 세밀한 특징 학습
- 특정 훈련 샘플의 noise까지 학습
→ 과적합
```

---

### Question 2: 큰 커널과 Receptive Field

#### Receptive Field 계산

```
단일 층:
kernel 3×3 → RF 3×3
kernel 5×5 → RF 5×5
kernel 7×7 → RF 7×7

다중 층 (3×3 반복):
층 1: RF 3×3
층 2: RF 5×5
층 3: RF 7×7
...
```

#### 왜 큰 커널 ≠ 더 좋은 성능?

**1. 파라미터 효율성**:

```
7×7 커널 1개:
- 49 파라미터
- RF: 7×7
- 비선형성: 1번

3×3 커널 3개:
- 27 파라미터 (더 적음!)
- RF: 7×7 (동일)
- 비선형성: 3번 (더 풍부)
```

**2. 특징 계층**:

```
큰 커널: 한 번에 큰 패턴 검출
        → 세밀한 특징 놓칠 수 있음

작은 커널 여러 층:
층 1: edge (선)
층 2: texture (질감)
층 3: part (부분)
→ 계층적 특징 학습
```

**3. MNIST 특성**:

```
MNIST 숫자 = 28×28 픽셀

7×7 커널:
- 숫자의 1/4을 한 번에 봄
- 너무 큰 맥락 → 지역적 특징 무시

3×3 커널:
- 획의 방향, 곡률 등 지역 특징 검출
- MNIST에 적합
```

#### 시각화

```
5 (숫자)를 인식할 때:

3×3 커널:
"위에 가로선" + "오른쪽에 곡선" + "아래 가로선"
→ 조합으로 5 인식

7×7 커널:
"전체 모양이 5처럼 생겼나?"
→ 변형에 약함 (기울어진 5 못 인식)
```

---

### Question 3: 더 깊거나 복잡한 데이터셋에서의 변화

#### MNIST vs CIFAR-10 vs ImageNet

| 데이터셋 | 크기 | 클래스 | 복잡도 |
|----------|------|--------|--------|
| MNIST | 28×28×1 | 10 | 낮음 |
| CIFAR-10 | 32×32×3 | 10 | 중간 |
| ImageNet | 224×224×3 | 1000 | 높음 |

#### 복잡한 데이터셋에서의 설계 변화

**채널 수**:
```
MNIST:    32-64 채널로 충분
CIFAR:    64-256 채널 필요
ImageNet: 64-2048 채널 (점진적 증가)
```

**커널 크기**:
```
MNIST:    3×3 일관
ImageNet:
  - 초기층: 7×7 (큰 RF로 저수준 특징)
  - 이후: 3×3 일관 (효율성)
```

**깊이**:
```
MNIST:    2-3 Conv층
CIFAR:    ~20층 (ResNet-20)
ImageNet: 50-152층 (ResNet-50/152)
```

#### 깊은 네트워크의 필수 기법

**1. Residual Connection**:
```python
# 기본
out = F(x)

# Residual
out = F(x) + x  # shortcut connection
```

**왜 필요?**
```
깊은 네트워크:
- 기울기 소실/폭발
- 최적화 어려움

Residual:
- 기울기가 shortcut으로 직접 전달
- 항등 함수(identity) 학습 쉬움
```

**2. Batch Normalization**:
```python
# 각 층의 출력을 정규화
x_norm = (x - mean) / std
out = gamma * x_norm + beta
```

**효과**:
```
- 내부 공분산 변화(Internal Covariate Shift) 감소
- 더 큰 학습률 사용 가능
- 정규화 효과 (약간의 과적합 방지)
```

---

## Challenge 8: Deep Network의 압축과 기하학

> **책 참조**: Chapter 4, Section 4.3 (Universal Geometric Principles), Exercise 4.3.2 (Learned Low-Dimensional Manifolds)

### 배경 지식: Representation Learning

#### 신경망이 하는 일

```
입력 공간 → [변환] → 특징 공간 → [분류기] → 출력

예: 이미지 (784D) → 특징 (64D) → 클래스 (10D)
```

**핵심 아이디어**:
- 입력 공간에서는 분류하기 어려움
- 좋은 특징 공간에서는 선형 분류기로 충분

#### 차원의 저주와 압축

**입력**: 고차원 (예: 784차원)
**실제 데이터**: 저차원 매니폴드 위에 존재

```
784D 공간에서 MNIST:

실제로 숫자 이미지가 차지하는 공간은
전체 784D의 극히 일부 (대략 10-20D 정도)

나머지는 "불가능한 이미지" (노이즈)
```

#### PCA로 차원 분석

```python
from sklearn.decomposition import PCA

# 각 층의 활성화에 PCA 적용
pca = PCA(n_components=100)
pca.fit(activations)

# 분산 설명 비율
explained_variance_ratio = pca.explained_variance_ratio_
cumsum = np.cumsum(explained_variance_ratio)

# 90% 분산을 설명하는 차원 수
intrinsic_dim = np.searchsorted(cumsum, 0.9) + 1
```

---

### Question 1: 깊이에 따른 Intrinsic Dimensionality 변화

#### 관찰: 층이 깊어질수록 유효 차원 감소

```
MNIST CNN 예시:

입력:     784D (28×28)
Conv1 후: ~100D (유효)
Conv2 후: ~50D (유효)
FC 전:    ~20D (유효)
출력 전:  ~10D (클래스 수!)
```

#### PCA 스펙트럼으로 확인

```
입력층:
특이값: σ₁ > σ₂ > σ₃ > ... > σ₁₀₀ (천천히 감소)
      ████████████████████████
      ███████████████████
      ██████████████████
      ... (많은 방향에 에너지 분산)

깊은 층:
특이값: σ₁ >> σ₂ >> σ₃ > σ₄ ≈ ... ≈ 0
      ████████████████████████
      ██████
      ██
      (소수 방향에 에너지 집중)
```

#### 계층적 추상화 해석

```
층 1: "모든 픽셀 정보" → 고차원
     어떤 특징이 중요한지 아직 모름

층 2: "가장자리와 질감" → 중차원
     불필요한 정보 일부 버림

층 3: "부분과 형태" → 저차원
     분류에 필요한 핵심 정보만 남김

출력: "어떤 숫자인가" → 10D
     최종 결정에 필요한 것만
```

#### 시각화

```
입력 공간 (784D):
  ?
 ???
?????      모든 방향으로 퍼짐
 ???
  ?

깊은 층 (10D에 가까움):

  ●●●     몇 개 방향으로만 퍼짐
  ●●●     (각 클래스 하나씩)

```

---

### Question 2: 비선형성 전후 PCA 스펙트럼 비교

#### ReLU의 효과

**ReLU 전 (선형 출력)**:
```
z = Wx + b
→ 선형 변환
→ 데이터의 분포 형태 유지 (회전, 스케일)
```

**ReLU 후**:
```
a = ReLU(z) = max(0, z)
→ 음수 부분이 0으로 접힘
→ 데이터가 양의 orthant로 "접힘"
```

#### 기하학적 변화

```
ReLU 전 (2D 예시):

    ↑
  ╲ │ ╱
   ╲│╱
────●────→    데이터가 모든 방향으로 퍼짐
   ╱│╲
  ╱ │ ╲


ReLU 후:

    ↑
    │ ╱
    │╱
────●────→    음수 방향이 접혀서
    │           양의 quadrant에 집중
    │
```

#### PCA 스펙트럼 변화

```
ReLU 전:
분산이 여러 방향에 분산
σ₁ ≈ σ₂ ≈ σ₃ ≈ ... (비교적 균등)

ReLU 후:
분산이 소수 방향에 집중
σ₁ >> σ₂ > σ₃ >> σ₄ ...

이유:
- 음수 → 0 변환으로 많은 변동이 사라짐
- 남은 변동은 특정 방향에 집중
```

#### Manifold 재형성

```
비선형성의 역할:
1. 데이터를 "접고 펴기"
2. 원래 분리 불가능한 클래스를 분리 가능하게 만듦

예:
입력: ●○●○●○ (XOR 패턴, 선형 분리 불가)

ReLU 적용 후:
    ●●●
   ╱
  ○○○     선형 분리 가능해짐!
```

---

### Question 3: 데이터 줄이면 압축 vs 암기?

#### 일반화 vs 암기

**충분한 데이터로 학습 (일반화)**:
```
학습: "3"이라는 숫자의 공통 특징 학습
- 위에 곡선
- 중간에 꺾임
- 아래에 곡선

새로운 "3" 등장 → 특징 매칭 → 올바른 분류
```

**적은 데이터로 학습 (암기)**:
```
학습: 각 훈련 샘플을 개별적으로 기억
- "이 특정 3"의 픽셀 패턴
- "저 특정 3"의 픽셀 패턴

새로운 "3" 등장 → 정확히 일치하는 것 없음 → 실패
```

#### PCA 스펙트럼으로 진단

**압축 (일반화)**:
```
σ₁ ████████████████████████
σ₂ ████████
σ₃ ████
σ₄ ██
σ₅ █
...
(급격히 감소)

해석:
- 소수의 "일반적 특징"에 정보 집중
- 나머지 방향은 노이즈로 무시
```

**암기 (과적합)**:
```
σ₁ ████████████
σ₂ ██████████
σ₃ █████████
σ₄ ████████
σ₅ ███████
...
(완만하게 감소)

해석:
- 많은 방향에 정보 분산
- 각 샘플을 구분하려면 많은 차원 필요
- "개별 샘플 ID"를 인코딩하는 것과 유사
```

#### 직관적 비유

```
일반화:
"모든 고양이의 공통점" 학습
→ 특징 공간: "털", "귀 모양", "눈" 정도 (저차원)
→ 새로운 고양이도 잘 분류

암기:
"훈련 고양이 각각의 사진" 학습
→ 특징 공간: "사진1의 픽셀", "사진2의 픽셀", ... (고차원)
→ 새로운 고양이 = 본 적 없음 → 실패
```

#### 실험으로 확인

```python
# 데이터 양에 따른 유효 차원 변화

for data_fraction in [0.01, 0.1, 0.5, 1.0]:
    model = train_cnn(data[:int(len(data)*data_fraction)])

    for layer in model.layers:
        activations = get_activations(layer, test_data)
        pca = PCA(n_components=50)
        pca.fit(activations)

        # 90% 분산 설명 차원
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.searchsorted(cumsum, 0.9) + 1

        print(f"Data: {data_fraction*100}%, Layer: {layer.name}, Dim: {intrinsic_dim}")

# 예상 결과:
# Data: 1%, Layer: FC, Dim: 45  ← 암기 (고차원)
# Data: 100%, Layer: FC, Dim: 12 ← 일반화 (저차원)
```

---

# 📌 Generative AI에서의 응용 (책에 없는 연결)

> 이 섹션은 책의 수학적 내용을 실제 Generative AI 모델과 연결합니다.

## Challenge 1-2 → VAE의 Latent Space

**SVD와 VAE의 관계**:
```
SVD:  X ≈ U_k Σ_k V_k^T   (k차원 근사)
VAE:  x → [Encoder] → z → [Decoder] → x'

둘 다 "저차원 표현"을 찾는다!

SVD:  선형 압축 (rotation-scale-rotation)
VAE:  비선형 압축 (neural network)

VAE의 z가 하는 일 = SVD의 U_k Σ_k 가 하는 일
"입력을 저차원으로 요약"
```

**LoRA (Low-Rank Adaptation)**:
```
LLM 파인튜닝에서:
W_new = W_original + ΔW
ΔW ≈ A B^T   (low-rank!)

rank-4면:
1000×1000 행렬 대신
1000×4 + 4×1000 = 8,000 파라미터만!

SVD에서 배운 "low-rank = 핵심 정보"가
여기서 "low-rank = 필요한 변화량"으로 적용됨
```

## Challenge 3 → Diffusion의 Score Matching

**Backprop과 Score Function**:
```
Diffusion Model:
∇_x log p(x) = "score function"
= "데이터가 있을 확률이 높은 방향"

학습 시:
∂L/∂θ = E[∥s_θ(x) - ∇_x log p(x)∥²]에 대한 기울기

Chain Rule이 여기서도 필수!
```

**Reverse Diffusion as ODE**:
```
dx/dt = -½ β(t)[x + ∇_x log p_t(x)]

이것은 ODE! (Challenge 4와 연결)

Neural ODE:
dx/dt = f_θ(x, t)

둘 다 "시간에 따른 변화"를 모델링
```

## Challenge 5-6 → Transformer 학습

**왜 Adam이 Transformer에 필수인가**:
```
Transformer의 파라미터:
- Attention weights: 수십만~수백만
- FFN weights: 더 많음
- 각 파라미터의 기울기 스케일이 다름

Adam의 역할:
1. 자주 업데이트되는 임베딩 → 작은 학습률
2. 드물게 업데이트되는 attention → 큰 학습률
→ 자동으로 적응!
```

**Learning Rate Warmup**:
```
왜 Transformer는 warmup이 필요한가?

초기 상태:
- 가중치가 random
- 기울기가 매우 크고 불안정
- Adam의 v_t가 아직 정확하지 않음

Warmup:
η = η_max × (step / warmup_steps)

처음엔 작게 시작 → 점점 증가
→ Adam이 안정화될 시간 확보
```

## Challenge 7-8 → Vision Transformer와 Representation

**CNN → ViT의 변화**:
```
CNN (Challenge 7):
- 지역적 특징 (3×3 커널)
- Inductive bias: "가까운 픽셀이 관련있다"
- 채널 = 특징 종류

ViT:
- 전역적 특징 (Self-Attention)
- No inductive bias: "모든 위치가 상호작용"
- 채널 = 토큰 차원

공통점:
- 층이 깊어질수록 유효 차원 감소 (Challenge 8)
- High-level 특징으로 수렴
```

**CLIP의 Representation Learning**:
```
CLIP = Image Encoder + Text Encoder

훈련 후:
- 이미지 특징 ∈ R^512
- 텍스트 특징 ∈ R^512
- 같은 공간에 정렬됨!

Challenge 8의 "압축" 관점:
고양이 이미지 (224×224×3 = 150,528D)
    ↓ [Encoder]
고양이 특징 (512D)
    ↓ [내적]
"a photo of a cat" 특징 (512D)

수십만 차원 → 512차원으로 압축
필수 정보만 남김
```

---

# 🎯 직관적 비유 모음 (책에 없는 설명)

## 1. SVD = 사진 압축의 수학적 원리

```
원본 사진: 1000×1000 = 백만 픽셀

SVD 후:
σ₁ = 10000 (하늘 배경)
σ₂ = 5000  (나무)
σ₃ = 2000  (사람)
σ₄ = 100   (먼지 같은 노이즈)
...

k=3만 유지하면:
- 저장: 3×(1000+1000) = 6,000 숫자
- 품질: 하늘+나무+사람 유지, 노이즈 제거
- 압축률: 166배!

JPEG도 비슷한 원리 (DCT 사용)
```

## 2. Backprop = 탓하기의 연쇄

```
시험 점수 나쁨 (Loss)
    ↓ 왜?
수학 문제 틀림 (출력 오차)
    ↓ 왜?
공식 외우기 실패 (중간층 오차)
    ↓ 왜?
수업시간 졸음 (입력층 오차)

각 단계에서:
"이 실수가 최종 결과에 얼마나 기여했나?"
= ∂L/∂(각 단계)

Backprop = "탓"을 거꾸로 전파
```

## 3. Optimizer = 등산 전략

```
산 정상 찾기 (최소점 = 골짜기 찾기)

SGD:
"눈 감고 가장 가파른 방향으로 내려간다"
→ 단순하지만, 구덩이에 빠지면 못 나옴

Momentum:
"관성을 가진 공처럼 굴러간다"
→ 얕은 구덩이는 통과, 속도 유지

Adam:
"스마트 등산가: 각 방향마다 적절한 보폭"
→ 가파른 곳: 조심히 (작은 보폭)
→ 평평한 곳: 빠르게 (큰 보폭)
```

## 4. CNN Channels = 전문가 팀

```
32 채널 CNN = 32명의 전문가

전문가 1: "가로선 탐지" 담당
전문가 2: "세로선 탐지" 담당
전문가 3: "곡선 탐지" 담당
...
전문가 32: "특정 질감 탐지" 담당

각자 전문 분야만 보고 의견 제시
→ 최종 분류기가 종합 판단

채널 늘리기 = 전문가 더 고용
→ 더 세밀하지만, 비용(파라미터) 증가
```

## 5. Representation Learning = 언어 통역

```
한국어 (입력) → [통역사] → 영어 (출력)
이미지 (입력) → [CNN] → 특징 (출력)

좋은 통역:
- 의미는 보존
- 불필요한 말투/사투리 제거
- 핵심만 전달

좋은 표현 학습:
- 클래스 정보는 보존
- 불필요한 픽셀 노이즈 제거
- 분류에 필요한 것만 전달
```

## 6. Dimensionality Reduction = 그림자

```
3D 물체 → 2D 그림자
1000D 데이터 → 10D 표현

그림자도 정보를 담는다:
- 물체의 외곽선
- 대략적인 형태
- 크기 비율

좋은 "그림자" (projection):
- 물체를 구분 가능하게 유지
- 불필요한 차원은 압축

PCA = "가장 정보를 보존하는 그림자 방향 찾기"
```

---

# 핵심 요약표

| Challenge | 핵심 개념 | 직관적 설명 | AI 응용 |
|-----------|-----------|-------------|---------|
| **1** | Einstein Notation | 반복 인덱스 = 합산 | GPU 텐서 연산 최적화 |
| **2** | SVD/PCA | 중요한 방향만 남기기 | 차원 축소, 압축, LoRA |
| **3** | Backpropagation | 끝에서부터 기울기 전파 | 모든 딥러닝 학습의 기초 |
| **4** | ODE 회귀 | 변화율 직접 피팅 | Neural ODE, 물리 시뮬레이션 |
| **5** | Convex vs Non-convex | 지형에 따른 수렴 차이 | 초기화 전략, 학습률 스케줄링 |
| **6** | Optimizers | 적응적 학습률 | Adam이 표준인 이유 |
| **7** | CNN 설계 | 채널/커널 선택 | 효율적인 네트워크 아키텍처 |
| **8** | Representation | 깊을수록 압축 | 특징 학습, 전이 학습 |

---

# 🔗 Generative AI 연결 요약

| 수학 개념 | Generative AI 응용 |
|-----------|-------------------|
| **SVD (Ch.1)** | LoRA 파인튜닝, 압축, PCA whitening |
| **Chain Rule (Ch.2)** | Score Matching, DDPM 학습 |
| **ODE (Ch.2)** | Neural ODE, Flow Matching, Diffusion SDE |
| **Optimization (Ch.3)** | Adam, Warmup, LR Scheduling |
| **CNN (Ch.4)** | U-Net (Diffusion), Vision Encoder |
| **Representation (Ch.4)** | Latent Space, CLIP embeddings |

---

# 추천 학습 순서

```
1. 선형대수 기초 (Ch.1)
   ├── 벡터/행렬 연산
   ├── SVD/Eigendecomposition
   └── Einstein Notation

2. 미적분 기초 (Ch.2)
   ├── Chain Rule
   ├── Jacobian
   └── ODE 기초

3. 최적화 (Ch.3)
   ├── Gradient Descent
   ├── Convex vs Non-convex
   └── Adam, SGD 비교

4. 딥러닝 (Ch.4)
   ├── CNN 구조
   ├── Backpropagation
   └── Representation Learning
```

---

*이 문서는 KAIST AI Mini-Course의 Challenge 문제에 대한 상세 해설입니다.*
*Mathematics of Generative AI Book (https://github.com/mchertkov/Mathematics-of-Generative-AI-Book) 참고*
