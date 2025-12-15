# Chapter 1: Linear Algebra (of AI)

> PDF pages 10-36 | 핵심: 벡터, 행렬, 텐서, 컨볼루션, Transformer

---

## 1.1 Foundations of Representing Data (데이터 표현의 기초)

### 1.1.1 Vector (벡터)

**정의**: n차원 실수 공간의 순서가 있는 숫자 모음
$$\mathbf{v} = [v_1, v_2, \ldots, v_n]^\top \in \mathbb{R}^n$$

**핵심 연산**:
| 연산 | 수식 | 설명 |
|------|------|------|
| 덧셈 | $(u+v)_i = u_i + v_i$ | 같은 차원의 벡터끼리 |
| 스칼라 곱 | $(cv)_i = cv_i$ | 스케일링 |
| 내적 | $uv^\top = \sum_{i=1}^n u_i v_i$ | 유사도 측정 |
| 노름 | $\|v\| = \sqrt{vv^\top}$ | 크기(길이) |

**AI 응용**: 이미지 픽셀 강도, 단어 임베딩, 그래디언트 방향 표현

### 1.1.2 Matrix (행렬) - 선형 변환의 표현

**정의**: 2D 숫자 배열 $A \in \mathbb{R}^{m \times n}$

**핵심 연산**:
| 연산 | 수식 | 설명 |
|------|------|------|
| 덧셈 | $(A+B)_{ij} = A_{ij} + B_{ij}$ | 원소별 덧셈 |
| Hadamard 곱 | $(A \odot B)_{ij} = A_{ij}B_{ij}$ | 원소별 곱셈 |
| 행렬 곱 | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | 선형 변환 합성 |
| 전치 | $(A^\top)_{ij} = A_{ji}$ | 행↔열 교환 |
| 역행렬 | $AA^{-1} = I$ | 변환의 역 |

**행렬 = 선형 변환**: 모든 선형 맵 $L: \mathbb{R}^n \to \mathbb{R}^m$은 행렬 $A$로 표현 가능

**선형 변환의 3가지 유형**:
1. **회전 (Rotation)**: z축 기준 θ 회전
   $$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

2. **스케일링 (Re-scaling)**: 축별 크기 조절
   $$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

3. **투영 (Projection)**: 부분공간으로 매핑
   - 예: x축으로 투영하면 y성분이 사라짐
   - 3D 물체를 2D 사진으로 찍는 것도 투영의 일종
   $$P_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

### 1.1.3 Tensor (텐서)

**어원**: 라틴어 *tendere* (늘리다) - "변형"을 나타내는 객체

**정의**: k차 텐서는 k개의 차원을 가진 배열
$$T \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$$

**랭크 (Rank)**:
| 랭크 | 이름 | 예시 |
|------|------|------|
| 0 | Scalar | 온도, 손실값 |
| 1 | Vector | 단어 임베딩 |
| 2 | Matrix | 이미지 (H×W) |
| 3+ | Higher-rank Tensor | RGB 이미지 (H×W×3), 비디오 (H×W×3×T) |

**ML 텐서 vs 수학 텐서**:
- **수학**: 좌표 변환 법칙을 따르는 다중선형 객체
- **ML (PyTorch/NumPy)**: 단순히 다차원 숫자 배열 ← 이 책에서 주로 사용

**텐서 연산**:
| 연산 | 설명 | 결과 |
|------|------|------|
| **Direct Product** (⊗) | 두 텐서 결합 | 랭크 증가 |
| **Contraction** | 공유 인덱스 합 | 랭크 감소 (행렬곱의 일반화) |

**Direct Product 예시**: $u \otimes v = A$, where $A_{ij} = u_i v_j$
```
u = [1, 2], v = [3, 4]
u ⊗ v = [[1·3, 1·4], [2·3, 2·4]] = [[3, 4], [6, 8]]
```

**Contraction 예시**: 행렬 곱 $C = AB$는 $C_{ij} = \sum_k A_{ik}B_{kj}$

---

## 1.2 Convolution (컨볼루션)

### 정의

**연속**: 두 함수가 서로 "밀어붙이며" 상호작용하는 연산
- 한 함수를 뒤집어서 다른 함수 위로 슬라이딩하며 겹치는 면적을 계산
$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau$$

**이산**: 벡터 $x \in \mathbb{R}^n$과 커널 $k \in \mathbb{R}^m$ (m < n)의 컨볼루션
$$y_i = \sum_{j=1}^{m} k_j \cdot x_{i+j-1}, \quad i = 1, 2, \ldots, n-m+1$$

### 직관: 슬라이딩 윈도우

컨볼루션은 커널을 입력 위로 "슬라이딩"하며 지역 패턴을 검출:
```
입력 x: [1, 2, 3, 0, 4, 5, 6, 1]
커널 k: [1, 0, -1]

y[0] = 1·1 + 2·0 + 3·(-1) = -2  (위치 0-2)
y[1] = 2·1 + 3·0 + 0·(-1) = 2   (위치 1-3)
...
```

### 컨볼루션 = 구조화된 선형 변환

컨볼루션은 특수한 형태의 행렬 곱으로 표현 가능:
$$y = A(k) \cdot x$$

여기서 $A(k)$는 Toeplitz 행렬 (대각선 방향으로 같은 값)

**왜 컨볼루션을 쓰는가?**
- 일반 행렬보다 **파라미터가 적음**: 3×3 커널 = 9개 파라미터로 전체 이미지 처리
- **이동 불변성**: 고양이가 왼쪽에 있든 오른쪽에 있든 같은 커널로 검출
- **계산 효율성**: FFT(고속 푸리에 변환)로 빠르게 계산 가능

### 2D 컨볼루션 (CNN의 기본)

이미지 $x \in \mathbb{R}^{n \times n}$과 필터 $k \in \mathbb{R}^{m \times m}$:
$$y_{i,j} = \sum_{h,w=1}^{m} k_{h,w} \cdot x_{i+h-1, j+w-1}$$

**예시** (PDF Example 1.1.2):
```
x = [[1,2,3,0],     k = [[1, 0],
     [4,5,6,1],          [0,-1]]
     [7,8,9,0],
     [1,0,2,3]]

y[0,0] = 1·1 + 2·0 + 4·0 + 5·(-1) = -4
y[0,1] = 2·1 + 3·0 + 5·0 + 6·(-1) = -4
...
결과: y = [[-4,-4,2], [-4,-4,6], [7,6,6]]
```

---

## 1.3 Matrix Decomposition (행렬 분해)

### 1.3.1 SVD (Singular Value Decomposition)

**정의**: 임의의 행렬 $X$를 세 행렬의 곱으로 분해
$$X = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: 왼쪽 특이벡터 (데이터 공간의 주요 방향)
- $\Sigma \in \mathbb{R}^{m \times n}$: 특이값 대각행렬 (각 방향의 중요도, 내림차순 정렬)
- $V^T \in \mathbb{R}^{n \times n}$: 오른쪽 특이벡터 (특징 공간의 주요 방향)

#### 직관 1: "루빅스 큐브 풀기"

SVD는 복잡하게 엉킨 데이터를 **가장 단순한 형태로 풀어내는** 것:
1. **$V^T$ (첫 번째 회전)**: 데이터를 "표준 자세"로 돌림 - 가장 변동이 큰 방향이 x축이 되도록
2. **$\Sigma$ (스케일링)**: 각 축 방향으로 늘이거나 줄임 - "이 방향이 얼마나 중요한지" 표시
3. **$U$ (두 번째 회전)**: 원래 공간으로 다시 돌려놓음

마치 루빅스 큐브를 풀 때 **표준 상태로 만든 다음, 필요한 조작을 하고, 다시 원래대로 돌려놓는** 것과 비슷!

#### 직관 2: "타원 → 원" 변환

데이터가 타원 모양으로 퍼져 있다면:
- **$V^T$**: 타원의 장축/단축을 x/y축에 맞춤
- **$\Sigma$**: 장축/단축의 길이 (얼마나 늘어났는지)
- **$U$**: 타원의 원래 방향

이걸 **역으로 적용**하면 타원이 원이 됨! (이게 Whitening)

#### Low-Rank Approximation: "핵심만 남기기"

$$X \approx U_k \Sigma_k V_k^T$$

상위 k개의 특이값만 사용하면:
- **노이즈 제거**: 작은 특이값 = 노이즈, 큰 특이값 = 신호
- **압축**: 원본 (m×n) → 압축 (m×k) + (k×k) + (k×n), k << min(m,n)
- **Matrix Completion**: Netflix 추천에서 "빈 칸 채우기"

**예시**: MNIST 손글씨
- 원본: 784차원 (28×28 픽셀)
- SVD 상위 50개만: 핵심 특징 유지하면서 93% 압축
- 상위 10개로도 숫자 구별 가능!

**AI 응용**:
- **PCA (주성분 분석)**: 차원 축소의 핵심
- **추천 시스템**: User-Item 행렬 인수분해
- **LoRA (Low-Rank Adaptation)**: LLM 파인튜닝 시 파라미터 효율화
- **이미지 압축**: 상위 k개 특이벡터만 저장

### 1.3.2 Eigen Decomposition (고유값 분해)

**정의**: 정방행렬 $A$에 대해
$$A = V \Lambda V^{-1}$$

- **고유값 $\lambda$**: $Av = \lambda v$를 만족하는 스칼라
- **고유벡터 $v$**: 변환해도 방향이 바뀌지 않는 벡터
- **대칭 행렬**: $A = V \Lambda V^T$ (직교 고유벡터를 가짐)

#### 직관: "뚝심 있는 벡터"

고유벡터는 **행렬에 의해 변환되어도 방향이 바뀌지 않는 특별한 벡터**:
- 대부분의 벡터: $A$를 곱하면 방향도 바뀌고 크기도 바뀜
- 고유벡터: $A$를 곱해도 **방향은 그대로**, 크기만 $\lambda$배로 변함

**비유**: 회전목마에서
- 일반 벡터: 빙글빙글 돌아감
- 고유벡터: 회전축 방향! 제자리에서 늘어나거나 줄어들기만 함

**왜 중요한가?**
- 고유벡터 = 행렬 $A$의 **"본질적인 방향"**
- 고유값 = 그 방향으로 **"얼마나 늘어나는지"**
- 행렬의 거동을 이해하려면 고유값/고유벡터를 보면 됨

#### SVD vs Eigen 관계

| 항목 | SVD | Eigen |
|------|-----|-------|
| 적용 대상 | 임의의 m×n 행렬 | 정방행렬 (n×n) |
| 분해 | $X = U\Sigma V^T$ | $A = V\Lambda V^{-1}$ |
| 직교성 | $U$, $V$ 항상 직교 | 대칭 행렬일 때만 직교 |

**핵심 연결**:
- $X^TX$의 고유벡터 = $X$의 오른쪽 특이벡터 $V$
- $XX^T$의 고유벡터 = $X$의 왼쪽 특이벡터 $U$
- $X^TX$의 고유값 = 특이값의 제곱 $\sigma_i^2$

### 1.3.3 Graph Laplacian (그래프 라플라시안)

그래프 구조를 행렬로 표현하고, 그 행렬의 **고유값 분해**로 그래프를 분석!

**정의**:
$$L = D - A$$

- $A$: 인접 행렬 (노드 i, j가 연결되어 있으면 $A_{ij} = 1$)
- $D$: 차수 행렬 (대각 원소 = 각 노드의 연결 수)
- $L$: 라플라시안 행렬

#### 직관: "용수철 침대"

그래프를 **용수철로 연결된 공들의 네트워크**로 상상:
- 각 노드 = 공
- 각 간선 = 용수철
- 공을 하나 들어올리면? → 연결된 공들도 따라 올라옴

**$L$의 의미**: "$L \cdot x$는 각 노드가 이웃들과 얼마나 다른지"를 측정
$$[Lx]_i = d_i x_i - \sum_{j \sim i} x_j = \sum_{j \sim i} (x_i - x_j)$$

노드 $i$의 값과 이웃들의 값 차이의 합!

#### Fiedler Vector: "그래프 자르기"

$L$의 고유값 분해:
- **가장 작은 고유값 $\lambda_1 = 0$**: 대응 고유벡터 = 상수 벡터 (모든 노드가 같은 값)
- **두 번째로 작은 고유값 $\lambda_2$**: **Fiedler value** (그래프가 얼마나 잘 연결되어 있는지)
- **두 번째 고유벡터**: **Fiedler vector** (그래프 클러스터링에 사용!)

**Fiedler Vector로 클러스터링**:
1. Fiedler vector의 각 원소 = 각 노드의 "좌표"
2. 양수 노드들 → 그룹 1
3. 음수 노드들 → 그룹 2
4. 자연스럽게 그래프가 두 부분으로 나뉨!

**예시**: 소셜 네트워크에서 커뮤니티 탐지
- Fiedler vector 값이 비슷한 사람들 = 같은 커뮤니티
- GNN (Graph Neural Network)에서 노드 임베딩의 기초

---

## 🔑 Data Whitening (화이트닝) - VAE/Diffusion 연결

**왜 중요한가**: SVD는 단순한 행렬 분해가 아니라, **찌그러진 타원 데이터를 동그란 공(표준정규분포)으로 만드는** 도구

### 직관적 이해: "찌그러진 사진 복구하기"

1. **원본 데이터**: 럭비공처럼 찌그러진 타원 형태 (어떤 축은 길고 어떤 축은 짧음)
2. **SVD로 축 찾기**: 데이터가 늘어난 방향(고유 축)과 얼마나 늘어났는지(특이값) 파악
3. **나누기(Normalization)**: 늘어난 만큼 다시 줄임 (σ=10이면 10으로 나눔)
4. **결과**: 모든 방향으로 균등하게 퍼진 동그란 공 → 표준정규분포

### 왜 "둥근 공"이어야 하나?

**빈틈 없애기**: 데이터가 흩어져 있으면 랜덤하게 찍었을 때 "꽝"이 나옴
- 뭉쳐 있어야 아무 점이나 찍어도 의미 있는 데이터 근처에 있음

**방향 무관성 (Isotropic)**: 동그란 공은 어느 방향으로 돌려도 같은 모양
- AI가 특정 방향에 편향되지 않고 학습할 수 있음

**수학적 편의**: 정규분포는 계산이 깔끔하고 미분 가능

### 수식

공분산 행렬이 타원의 모양을 결정:
$$C = \frac{1}{n}X^TX = V\Sigma^2 V^T$$

Whitening으로 타원 → 원:
$$X_{white} = X V \Sigma^{-1}$$

### 생성 AI에서의 역할: "점토 빚기"

#### VAE: "도서관" 비유

VAE의 잠재 공간을 **도서관**이라고 상상하자:
- **Encoder**: 이미지(책)를 받아서 → 좌표(책 번호)로 변환
- **Latent Space**: 모든 책이 정리된 도서관
- **Decoder**: 좌표를 받아서 → 다시 이미지를 복원

**문제**: 도서관이 엉망이면?
- 책이 한쪽 구석에만 쌓여 있음 (데이터 뭉침)
- 중간에 빈 공간이 많음 (구멍)
- 랜덤한 위치에서 책을 찾으면 → 아무것도 없음!

**KL Divergence = "잔소리꾼 사서장"**

KL Divergence 손실 항이 하는 일:
- "책들을 동그랗게 배치해!" (표준정규분포로 강제)
- "너무 한 곳에 몰리면 안 돼!" (분산 유지)
- "원점 근처에 있어야 해!" (평균 = 0으로)

$$\mathcal{L}_{KL} = D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

결과: 잠재 공간이 **동그란 공** 모양이 됨!
- 아무 점이나 찍어도 의미 있는 이미지 생성 가능
- 두 이미지 사이를 보간하면 자연스러운 중간 이미지

#### Diffusion: "녹였다가 다시 얼리기"

Diffusion 모델의 핵심 아이디어:
1. **Forward Process (녹이기)**: 이미지에 조금씩 노이즈를 더함
   - 처음: 선명한 이미지
   - 마지막: 순수한 노이즈 (표준정규분포 = 동그란 공!)

2. **Reverse Process (얼리기)**: 노이즈에서 조금씩 이미지를 복원
   - 신경망이 "지금 단계에서 노이즈가 뭔지" 예측
   - 그 노이즈를 빼면서 점점 선명해짐

**왜 이게 Whitening과 연결되나?**
- Forward의 최종 상태 = 완전히 whitened된 데이터 (모든 구조 사라짐)
- Reverse = whitened → 원래 데이터로 "색칠"하는 과정

| 모델 | Whitening 역할 |
|------|----------------|
| **VAE** | KL Divergence가 잠재 공간을 동그란 공으로 강제 |
| **Diffusion** | Forward가 데이터를 노이즈(동그란 공)로 녹임 |
| **Normalizing Flow** | 복잡한 분포를 뒤틀어서 가우시안으로 변환 |

**핵심**: AI에게 표준정규분포(동그란 공)는 **"가장 깨끗한 빈 도화지"**
- 여기서 시작해야 무엇이든 그릴 수 있음
- 모든 방향이 동등 → 편향 없이 학습 가능

---

## 1.4 Applications in Generative AI (생성 AI 응용)

### 1.4.1 Diffusion Models의 행렬 변환

Diffusion 모델에서 데이터는 벡터로 표현되고, 반복적인 행렬 변환을 통해 노이즈 → 이미지로 정제:

$$x_{t+1} = A_t x_t + b_t, \quad t = 0, \ldots, T$$

- $x_0$: 순수 노이즈 (초기 입력)
- $x_T$: 최종 생성 이미지
- $A_t$: 변환 행렬 (학습됨)
- $b_t$: 바이어스 항

**핵심**: $A_t$와 $b_t$는 $x_t$와 $t$의 비선형 함수로, 신경망이 학습

### 1.4.2 Transformer 메커니즘 (상세)

Transformer는 2017년 Vaswani et al.의 "Attention is All You Need"에서 소개됨.
**RNN/LSTM의 순차 처리 한계를 극복**하고, 병렬 처리로 긴 시퀀스를 효율적으로 다룸.

#### 핵심 아이디어: "모든 단어가 모든 단어를 본다"

기존 RNN: 단어를 하나씩 순서대로 처리 → 긴 문장에서 앞부분 정보 손실
Transformer: **모든 단어 쌍의 관계**를 동시에 계산 → 멀리 떨어진 단어도 직접 연결!

#### 토큰 임베딩 + 위치 인코딩

입력 시퀀스: $X = \{t_1, t_2, \ldots, t_n\} \in \mathbb{R}^{n \times d}$
- $n$: 시퀀스 길이 (예: 문장의 단어 수)
- $d$: 임베딩 차원 (예: 768, 1024)
- $t_i \in \mathbb{R}^d$: i번째 토큰의 임베딩 벡터

**위치 인코딩 (Positional Encoding)**: Transformer는 순서를 모르므로 위치 정보를 더해줌
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

#### Self-Attention: "누구에게 주목할까?"

**비유**: 회의에서 발언할 때
- **Query (Q)**: "내가 지금 궁금한 것" (질문)
- **Key (K)**: "각 사람이 가진 정보의 종류" (전문 분야)
- **Value (V)**: "각 사람이 실제로 가진 정보" (답변 내용)

Q와 K를 비교해서 "누구 말을 들을지" 결정 → 그 사람의 V를 가져옴!

**Step 1**: Query, Key, Value 계산
$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

- $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$: 학습된 가중치 행렬
- 같은 입력 $X$에서 세 가지 다른 "관점"을 추출

**Step 2**: Attention Score 계산
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

- $QK^\top$: 각 Query-Key 쌍의 유사도 (내적)
- $\sqrt{d}$로 나눔: 차원이 크면 내적 값이 커지므로 스케일 조정
- softmax: 가중치 합이 1이 되도록 정규화
- 결과: 각 토큰이 다른 토큰들의 **가중 평균**이 됨

**왜 $\sqrt{d}$로 나누나?**
- 내적 $q \cdot k = \sum q_i k_i$
- d가 크면 합이 커짐 → softmax가 극단적 (0 or 1에 가까움)
- 나눠주면 분산이 1로 유지 → 부드러운 attention

#### Multi-Head Attention: "다양한 관점으로 보기"

단일 attention = 하나의 "관점"으로만 관계 파악
Multi-Head = **여러 관점**을 동시에!

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

여기서 각 head는:
$$\text{head}_h = \text{Attention}(Q W_Q^h, K W_K^h, V W_V^h)$$

**왜 Multi-Head?**
- head 1: 문법적 관계 ("주어-동사")
- head 2: 의미적 관계 ("고양이-귀엽다")
- head 3: 지시어 관계 ("그것"이 가리키는 대상)
- 각 head가 서로 다른 패턴을 학습!

**예시** (GPT-2, 12 heads):
```
입력: "The cat sat on the mat because it was tired"
head 3이 학습한 것: "it" → "cat" (지시어 해결)
head 7이 학습한 것: "sat" → "cat" (주어-동사)
```

#### Feed-Forward Network (FFN): "정보 처리"

Attention 후 각 토큰을 독립적으로 변환:
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{d \times 4d}$: 차원 확장 (보통 4배)
- $W_2 \in \mathbb{R}^{4d \times d}$: 다시 원래 차원으로
- 중간에 비선형 활성화 (GELU)

#### GELU vs ReLU: "디머 스위치 vs 온오프 스위치"

**ReLU**: $\text{ReLU}(x) = \max(0, x)$
- 음수 → 0, 양수 → 그대로
- **온오프 스위치**: 꺼지면 완전히 0

**GELU**: $\text{GELU}(x) = x \cdot \Phi(x)$
- $\Phi(x)$: 표준정규분포의 CDF
- **디머 스위치**: 점진적으로 밝기 조절

**왜 Transformer는 GELU를 쓰나?**
1. **부드러운 그래디언트**: ReLU는 0에서 꺾이지만, GELU는 매끈함
2. **확률적 해석**: "이 뉴런이 활성화될 확률"에 비례해서 값을 스케일
3. **성능**: 실험적으로 NLP 태스크에서 더 좋은 결과

```
x = -2: ReLU=0,      GELU≈-0.045 (거의 0이지만 완전히 0은 아님)
x = -1: ReLU=0,      GELU≈-0.16
x =  0: ReLU=0,      GELU=0
x =  1: ReLU=1,      GELU≈0.84
x =  2: ReLU=2,      GELU≈1.96
```

#### Layer Normalization: "안정화"

각 층의 출력을 정규화:
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- $\mu$, $\sigma^2$: 해당 층의 평균/분산
- $\gamma$, $\beta$: 학습 가능한 스케일/시프트

**Batch Norm vs Layer Norm**:
- BatchNorm: 배치 내 같은 뉴런들끼리 정규화 (CNN에서 주로 사용)
- LayerNorm: 한 샘플 내 모든 뉴런을 정규화 (Transformer에서 사용)
- LayerNorm은 배치 크기에 무관 → 시퀀스 길이가 다양한 NLP에 적합

#### Residual Connection: "고속도로"

$$x_{out} = x_{in} + \text{Sublayer}(x_{in})$$

- 입력을 직접 출력에 더함 → 그래디언트가 깊은 층까지 잘 전달
- "새로 배운 것을 기존 지식에 더한다"

#### 전체 Transformer Block 구조

```
Input
  │
  ├─────────────────────┐
  ↓                     │
Multi-Head Attention    │ (Residual)
  ↓                     │
  + ←───────────────────┘
  ↓
Layer Norm
  │
  ├─────────────────────┐
  ↓                     │
Feed-Forward (GELU)     │ (Residual)
  ↓                     │
  + ←───────────────────┘
  ↓
Layer Norm
  ↓
Output
```

#### 최종 토큰 예측 (언어 모델)

마지막 층의 출력에서 vocabulary 확률 계산:
$$p(t_{n+1} | X) = \text{softmax}(W_{out} \hat{t}_{n+1} + b_{out})$$

- $W_{out} \in \mathbb{R}^{d \times |V|}$: vocabulary 크기만큼의 출력
- softmax: 각 단어가 다음에 나올 확률

---

## 연습문제 (PDF에서 발췌)

### Exercise 1.1.1: Matrix Multiplication and Elliptical Dynamics
초기 벡터 $x_0 = (a, 0)^\top$에 행렬 $A$를 반복 적용하여 타원 위의 점들을 생성하는 $A$ 설계하기.
- (a) 반장축 비율 $a/b$인 타원 위에 $x_t = A^t x_0$ 놓기
- (b) $t \to \infty$일 때 타원 전체를 덮도록 설계 가능한가?

### Exercise 1.1.2: Tensor Representation
- 비디오: 4D 텐서 (1920×1080, 30fps, 10초 → 300개 프레임)
- NLP 임베딩: 2D 텐서 (D차원 임베딩 × L개 토큰)

### Exercise 1.1.3: Einstein Summation
- 행렬-벡터 곱 $y = Ax$를 ESN으로: `ij,j->i`
- Frobenius 노름을 ESN으로: `ij,ij->`

### Exercise 1.1.4: Multi-Head Attention 계산 (PDF 상세 예제)
3×2 입력 행렬과 2개 head로 attention 계산 연습

---

## 노트북 가이드

### [MNIST-SVD.ipynb](LinearAlgebra/MNIST-SVD.ipynb)
MNIST 손글씨 데이터에 SVD 적용

**배우는 것**:
- 이미지 배치를 행렬로 변환 (100×784)
- 평균 이미지와 "고유숫자(eigendigit)" 시각화
- Rank-k 재구성으로 압축 효과 확인
- 특이값 스펙트럼 분석 (에너지 집중도)

**핵심 코드**:
```python
X_centered = X - mean_image
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# Rank-k 재구성
x_reconstructed = mean_image + (x @ Vt[:k].T) @ Vt[:k]
```

### [Ex1-1-1image.ipynb](LinearAlgebra/Ex1-1-1image.ipynb)
이미지의 텐서 표현 기초 (PDF Example 1.1.3 구현)

**배우는 것**:
- RGB 이미지: $T \in \mathbb{R}^{H \times W \times 3}$
- 채널 추출: `T[:, :, 1]` (Green 채널)
- 평균 강도 계산: 모든 픽셀의 평균
- Grayscale → RGB 변환 (colormap 적용)

---

## 다음 장과의 연결

| 연결 | 설명 |
|------|------|
| **→ Chapter 2** | Jacobian, 자동 미분의 심화 (Forward/Reverse AD) |
| **→ Chapter 3** | Gradient Descent에서 행렬 연산 활용 |
| **→ Chapter 4** | CNN, ResNet 구현 시 컨볼루션 적용 |
| **→ Chapter 5** | 공분산 행렬, 가우시안 분포와 SVD 연결 |
| **→ Chapter 6** | VAE의 잠재 공간이 왜 $\mathcal{N}(0,I)$를 목표로 하는지 |
| **→ Chapter 7** | Diffusion의 forward process = 점진적 whitening |

---

## 핵심 요약

| 개념 | 정의 | 직관 | AI 응용 |
|------|------|------|---------|
| Vector | $v \in \mathbb{R}^n$ | 방향+크기 | 임베딩, 그래디언트 |
| Matrix | $A \in \mathbb{R}^{m \times n}$ | 선형 변환 (회전/스케일/투영) | 가중치, 변환 |
| Tensor | n차원 배열 | 다차원 데이터 | 이미지, 비디오, 배치 |
| Convolution | 커널과 입력의 슬라이딩 내적 | 슬라이딩 윈도우 | CNN |
| SVD | $X=U\Sigma V^T$ | 루빅스 큐브 풀기, 타원→원 | PCA, LoRA, Matrix Completion |
| Eigen | $Av = \lambda v$ | 뚝심 있는 벡터 (방향 불변) | 공분산 분석, Graph 분석 |
| Graph Laplacian | $L = D - A$ | 용수철 침대 | GNN, 클러스터링 (Fiedler) |
| Whitening | $X V \Sigma^{-1}$ | 찌그러진 사진 복구 | VAE, Diffusion 전처리 |
| KL Divergence | $D_{KL}(q \| p)$ | 잔소리꾼 사서장 | VAE 정규화 |
| Attention | softmax(QKᵀ/√d)V | 누구 말을 들을까? | Transformer |
| GELU | $x \cdot \Phi(x)$ | 디머 스위치 (점진적 조절) | Transformer 활성화 |

---

## 참고 자료

- **PDF 원본**: MathGenAIBook12_14_25.pdf, pages 10-36
- **Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/
- **원논문**: Vaswani et al., "Attention is All You Need" (2017)
