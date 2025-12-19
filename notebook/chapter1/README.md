# Chapter 1: Linear Algebra (of AI)

> **PDF pages 10-36** | 핵심: 벡터, 행렬, 텐서, 컨볼루션, SVD, Transformer
>
> **KAIST Challenge 1-2 참조**: [KAIST-challenges-solutions.md](KAIST-challenges-solutions.md)

---

## 목차
1. [데이터 표현의 기초](#11-foundations-of-representing-data-데이터-표현의-기초)
2. [컨볼루션](#12-convolution-컨볼루션)
3. [행렬 분해](#13-matrix-decomposition-행렬-분해)
4. [생성 AI 응용](#14-applications-in-generative-ai-생성-ai-응용)
5. [연습문제와 풀이](#exercises-연습문제)

---

## 1.1 Foundations of Representing Data (데이터 표현의 기초)

### 1.1.1 Vector (벡터)

#### 정의 (책 원문)

> A vector is an ordered collection of numbers that can represent points, directions, or quantities in space.

**수학적 정의**: n차원 실수 공간의 순서가 있는 숫자 모음
$$\mathbf{v} = [v_1, v_2, \ldots, v_n]^\top \in \mathbb{R}^n$$

여기서:
- $v_i$: 벡터의 성분 (component)
- $n$: 벡터의 차원 (dimensionality)
- $[\cdot]$: 행벡터 표기
- ${}^\top$: 전치 (행벡터 → 열벡터)

#### 직관적 이해

**비유**: 데이터를 숫자로 나열한 리스트

```
사람의 특징 = [키(cm), 몸무게(kg), 나이]
           = [175, 70, 25]  ← 3차원 벡터

MNIST 이미지 = [pixel_1, pixel_2, ..., pixel_784]  ← 784차원 벡터
```

AI에게는 이미지, 단어, 소리 모두 **숫자 리스트(벡터)**로 보입니다.

#### 핵심 연산 (Key Operations)

| 연산 | 수식 (책 원문) | 직관적 설명 |
|------|---------------|-------------|
| **덧셈** | $(\mathbf{u} + \mathbf{v})_i = u_i + v_i$ | 같은 위치끼리 더하기 |
| **스칼라 곱** | $(c\mathbf{v})_i = cv_i$ | 벡터 크기를 c배로 |
| **내적 (Product)** | $\mathbf{u}\mathbf{v}^\top = \sum_{i=1}^n u_i v_i$ | 두 벡터의 유사도 측정 |
| **노름 (Norm)** | $\|\mathbf{v}\| = \sqrt{\mathbf{v}\mathbf{v}^\top}$ | 벡터의 길이 |

#### 내적의 기하학적 의미

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

- $\theta = 0°$: 같은 방향 → 내적 최대 (양수)
- $\theta = 90°$: 직교 → 내적 = 0
- $\theta = 180°$: 반대 방향 → 내적 최소 (음수)

**AI 응용**:
- **Word2Vec**: 단어 임베딩 벡터의 내적 = 의미적 유사도
- **Attention**: Query와 Key의 내적 = 관련성 점수

---

### 1.1.2 Matrix (행렬) - 선형 변환의 표현

#### 정의 (책 원문)

> A matrix is a 2D array of numbers that generalizes vectors to multiple dimensions.

$$A = \begin{pmatrix}
A_{11} & A_{12} & \cdots & A_{1n} \\
A_{21} & A_{22} & \cdots & A_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m1} & A_{m2} & \cdots & A_{mn}
\end{pmatrix} \in \mathbb{R}^{m \times n}$$

#### 핵심 연산 (책 원문 수식)

| 연산 | 수식 | 의미 |
|------|------|------|
| **덧셈** | $(A + B)_{ij} = A_{ij} + B_{ij}$ | 원소별 덧셈 |
| **Hadamard 곱** | $(A \odot B)_{ij} = A_{ij} B_{ij}$ | 원소별 곱셈 |
| **행렬 곱** | $(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$ | 변환의 합성 |
| **행렬-벡터 곱** | $(A\mathbf{v})_i = \sum_{j=1}^n A_{ij} v_j$ | 벡터 변환 |
| **전치** | $(A^\top)_{ij} = A_{ji}$ | 행↔열 교환 |
| **역행렬** | $AA^{-1} = A^{-1}A = I$ | 변환 취소 |

#### 행렬 = 선형 변환 (Linear Transformation)

> Any linear map $L : \mathbb{R}^n \to \mathbb{R}^m$ can be uniquely expressed in terms of a matrix $A \in \mathbb{R}^{m \times n}$.

**책의 핵심 예시**:

**1. 회전 (Rotation)** - z축 기준 θ만큼 회전:
$$R_z(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{pmatrix}$$

**2. 스케일링 (Re-scaling)**:
$$S = \begin{pmatrix}
s_x & 0 \\
0 & s_y
\end{pmatrix} \quad \Rightarrow \quad S\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} s_x x \\ s_y y \end{pmatrix}$$

**3. 투영 (Projection)** - x축으로 투영:
$$P_1 = \begin{pmatrix}
1 & 0 \\
0 & 0
\end{pmatrix} \quad \Rightarrow \quad P_1\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x \\ 0 \end{pmatrix}$$

#### 직관: "공간을 변형시키는 기계"

```
원본 격자:          행렬 A 적용 후:
□ □ □ □             ◇ ◇ ◇ ◇
□ □ □ □      →      ◇ ◇ ◇ ◇
□ □ □ □             ◇ ◇ ◇ ◇

행렬은 격자를 회전, 늘리기, 찌그러뜨리기 가능
```

---

### 1.1.3 Tensor (텐서) - 다차원으로의 일반화

#### 정의 (책 원문)

> A tensor $T$ of rank $k$ can be represented as:
> $$T \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$$
> where $d_i$ is the size along the $i$-th dimension.

#### 랭크 (Rank) = 차원의 수

| 랭크 | 이름 | 예시 | AI 응용 |
|------|------|------|---------|
| 0 | Scalar | 온도, Loss 값 | 손실 함수 출력 |
| 1 | Vector | 소리 파형, 단어 임베딩 | Word2Vec |
| 2 | Matrix | 흑백 이미지 (H×W) | 가중치 행렬 |
| 3 | 3-Tensor | 컬러 이미지 (H×W×3) | CNN 입력 |
| 4 | 4-Tensor | 비디오 배치 (B×H×W×C) | 배치 학습 |

#### 책 예시: RGB 이미지
$$T \in \mathbb{R}^{H \times W \times 3}$$
- H: 높이 (행 수)
- W: 너비 (열 수)
- 3: RGB 채널

#### 텐서 연산 (책 원문)

**1. Direct Product (텐서 곱) ⊗**

두 텐서를 결합하여 더 높은 랭크 생성:
$$(T_1 \otimes T_2)_{i_1,\ldots,i_{k+m}} = (T_1)_{i_1,\ldots,i_k} \cdot (T_2)_{i_{k+1},\ldots,i_{k+m}}$$

**예시 (책)**:
$$\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$

$$\mathbf{u} \otimes \mathbf{v} = \begin{pmatrix} 1 \cdot 3 & 1 \cdot 4 \\ 2 \cdot 3 & 2 \cdot 4 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 6 & 8 \end{pmatrix}$$

**2. Contraction (축약)**

공유된 인덱스를 따라 합산하여 랭크 감소:
$$T_{i_1,i_2,i_4} = \sum_{i_3=1}^{d_3} (T_1)_{i_1,i_2,i_3} \cdot (T_2)_{i_3,i_4}$$

**핵심**: 행렬 곱 $(AB)_{ij} = \sum_k A_{ik}B_{kj}$도 축약의 특수 케이스!

---

### 1.1.4 Einstein Summation Notation (아인슈타인 표기법)

> **KAIST Challenge 1** 직접 연결

#### 규칙

**반복되는 인덱스 = 자동으로 합산**

| 표기 | 수식 |
|------|------|
| 일반 표기 | $y_i = \sum_j A_{ij} x_j$ |
| Einstein | $y_i = A_{ij} x_j$ (j가 두 번 → 합산) |

#### 책 Exercise 1.1.3 풀이

**(1) 행렬-벡터 곱** $\mathbf{y} = A\mathbf{x}$:
$$y_i = A_{ij} x_j$$

**(2) Frobenius 노름** $\|A\|_F^2 = \sum_{i,j} A_{ij}^2$:
$$\|A\|_F^2 = A_{ij} A_{ij}$$

#### KAIST Challenge 1 - Question 1

> Given $C_{ikl} = A_{ij}B_{jkl}$, what is the dimensionality of C?

**풀이**:
- A: 인덱스 (i, j) → 2D
- B: 인덱스 (j, k, l) → 3D
- j가 양쪽에서 반복 → **합산되어 사라짐**
- 남는 인덱스: i, k, l

**답: C는 3차원 텐서 (I × K × L)**

#### KAIST Challenge 1 - Question 2

> Why is memory contiguity more important than FLOPs in GPU programming?

**핵심 답변**:

1. **GPU 메모리 계층**:
   ```
   레지스터 (빠름) → 공유 메모리 → L1/L2 캐시 → 글로벌 메모리 (느림)
   ```

2. **Coalesced Access (연속 접근)**:
   - 연속 메모리: 한 번에 128바이트 읽기
   - 비연속 메모리: 매번 새로운 요청 → **100배 느림**

3. **비유**:
   ```
   도서관에서 책 10권 빌리기:
   - 연속: 한 책장에서 쭉 10권 (1번 이동)
   - 비연속: 10개 책장 방문 (10번 이동)
   ```

---

## 1.2 Convolution (컨볼루션)

### 정의 (책 원문)

> Convolution combines two inputs $f$ and $g$, producing an output that reflects their interaction.

**1D 이산 컨볼루션** (벡터 $\mathbf{x} \in \mathbb{R}^n$, 커널 $\mathbf{k} \in \mathbb{R}^m$):
$$y_i = \sum_{j=1}^m k_j \cdot x_{i+j-1}, \quad i = 1, 2, \ldots, n - m + 1$$

### 컨볼루션 = 구조화된 행렬 곱

$$\mathbf{y} = A(\mathbf{k}) \cdot \mathbf{x}$$

여기서 $A(\mathbf{k})$는 **Toeplitz 행렬**:
$$A(\mathbf{k}) = \begin{pmatrix}
k_1 & 0 & \cdots & 0 \\
k_2 & k_1 & \cdots & 0 \\
\vdots & k_2 & \ddots & \vdots \\
k_m & \vdots & \cdots & k_1 \\
0 & k_m & \ddots & \vdots \\
\vdots & 0 & \cdots & k_m
\end{pmatrix}$$

### 2D 컨볼루션 (CNN의 기본)

**책 공식**:
$$y_{i,j} = \sum_{h,w=1}^m k_{h,w} \cdot x_{i+h-1,j+w-1}$$

### 책 Example 1.1.2 - 상세 계산

$$\mathbf{x} = \begin{pmatrix}
1 & 2 & 3 & 0 \\
4 & 5 & 6 & 1 \\
7 & 8 & 9 & 0 \\
1 & 0 & 2 & 3
\end{pmatrix}, \quad \mathbf{k} = \begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}$$

**계산 과정**:

| 위치 | 계산 | 결과 |
|------|------|------|
| $y_{0,0}$ | $1 \times 1 + 2 \times 0 + 4 \times 0 + 5 \times (-1)$ | $-4$ |
| $y_{0,1}$ | $2 \times 1 + 3 \times 0 + 5 \times 0 + 6 \times (-1)$ | $-4$ |
| $y_{0,2}$ | $3 \times 1 + 0 \times 0 + 6 \times 0 + 1 \times (-1)$ | $2$ |

**결과**:
$$\mathbf{y} = \begin{pmatrix}
-4 & -4 & 2 \\
-4 & -4 & 6 \\
7 & 6 & 6
\end{pmatrix}$$

### 왜 컨볼루션을 쓰는가?

| 장점 | 설명 |
|------|------|
| **파라미터 효율** | 3×3 커널 = 9개 파라미터로 전체 이미지 처리 |
| **이동 불변성** | 고양이가 어디 있든 같은 필터로 검출 |
| **계층적 특징** | 작은 패턴 → 큰 패턴 점진적 학습 |

---

## 1.3 Matrix Decomposition (행렬 분해)

### 1.3.1 SVD (Singular Value Decomposition)

#### 정의 (책 원문)

> SVD is a powerful linear algebra technique that can be applied to a batch of data points to extract a reduced-dimensional representation.

$$X = USV^\top$$

여기서:
- $U \in \mathbb{R}^{n \times n}$: 왼쪽 특이벡터 (행 공간의 기저)
- $S \in \mathbb{R}^{n \times d}$: 특이값 대각행렬 ($\sigma_1 \geq \sigma_2 \geq \cdots$)
- $V \in \mathbb{R}^{d \times d}$: 오른쪽 특이벡터 (열 공간의 기저)

#### 핵심 성질 (책 원문)

> The columns of $V$ (right singular vectors) are the eigenvectors of $X^\top X$, and the squares of the singular values $\sigma_i^2$ are the eigenvalues of $X^\top X$:

$$X^\top X = V S^\top S V^\top = V \, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2) \, V^\top$$

#### 직관 1: "회전 → 스케일 → 회전"

```
원본 데이터 (단위원)     V^T (회전)      Σ (스케일)      U (회전)
       ○          →        ○       →      ⬭       →      ⬬
```

어떤 선형 변환이든 **회전-스케일-회전**으로 분해 가능!

#### 직관 2: "타원 → 원" (Whitening 연결)

데이터가 타원 형태로 퍼져 있을 때:
- $V^\top$: 타원의 주축을 좌표축에 맞춤
- $\Sigma$: 각 축의 늘어난 정도
- 역으로 적용하면 **타원 → 원** (Whitening!)

#### Low-Rank Approximation

$$X \approx U_k \Sigma_k V_k^\top$$

상위 k개 특이값만 사용:
- **노이즈 제거**: 작은 특이값 = 노이즈
- **압축**: 저장 공간 절약
- **Matrix Completion**: Netflix 추천 시스템

#### KAIST Challenge 2 연결

> **Exercise**: 90% 에너지를 유지하는 최소 k 찾기

**총 에너지**: $E_{total} = \sum_{i=1}^{r} \sigma_i^2 = \|X\|_F^2$

**90% 조건**:
$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \geq 0.90$$

**Question 1**: k 선택 = 메모리 vs 정보 손실 트레이드오프
- k 작음: 메모리 적음, 정보 손실 큼
- k 큼: 메모리 많음, 정보 손실 적음

**Question 2**: 왜 $X^\top X$ 고유분해가 효율적인가?

| 방법 | 행렬 크기 | N=100,000, D=100 |
|------|-----------|------------------|
| X 직접 SVD | N×D | 80GB 메모리 |
| $X^\top X$ 고유분해 | D×D | 80KB 메모리 |

**이유**: PCA에서는 V(주성분)만 필요, U(각 샘플 좌표)는 불필요!

---

### 1.3.2 Eigen-Decomposition (고유값 분해)

#### 정의 (책 원문)

> An eigenvalue-eigenvector pair $(\lambda, \mathbf{v})$ satisfies:
> $$A\mathbf{v} = \lambda \mathbf{v}$$

정방행렬 $A$의 분해:
$$A = Q \Lambda Q^{-1}$$

- $Q$: 고유벡터 행렬
- $\Lambda$: 고유값 대각행렬

#### 직관: "뚝심 있는 벡터"

대부분의 벡터: $A$를 곱하면 방향도 바뀌고 크기도 바뀜
고유벡터: $A$를 곱해도 **방향은 그대로**, 크기만 $\lambda$배

```
일반 벡터:      고유벡터:
    →              →
  ↗   (A 적용)    ════→ (A 적용: 방향 유지)
```

#### SVD와 Eigen의 연결 (책 원문)

> For symmetric positive-definite matrices, such as the covariance matrix $\Sigma = \frac{1}{n} X^\top X$:

$$\Sigma = V \Lambda V^\top = V S^2 V^\top$$

| 항목 | SVD | Eigen |
|------|-----|-------|
| 적용 대상 | 임의의 m×n 행렬 | 정방행렬 (n×n) |
| 분해 | $X = U\Sigma V^\top$ | $A = Q\Lambda Q^{-1}$ |
| 직교성 | U, V 항상 직교 | 대칭일 때만 직교 |

**핵심 연결**:
- $X^\top X$의 고유벡터 = X의 오른쪽 특이벡터 V
- $XX^\top$의 고유벡터 = X의 왼쪽 특이벡터 U

---

### 1.3.3 Reduced Representation (차원 축소)

#### 책 원문

> To reduce the dimensionality of a single point $\mathbf{x}_i$, we project it onto the top $k$ singular directions:

$$\mathbf{z}_i = \mathbf{x}_i V_k$$

- $V_k \in \mathbb{R}^{d \times k}$: 상위 k개 특이벡터
- $\mathbf{z}_i \in \mathbb{R}^k$: 축소된 표현

#### 예시: MNIST

```python
# 원본: 784차원 (28×28)
# SVD 후 상위 50개 → 50차원
# 압축률: 93%
# 숫자 구별: 여전히 가능!
```

---

## 1.4 Applications in Generative AI (생성 AI 응용)

### 1.4.1 Diffusion Models의 행렬 변환

#### 책 원문

> In generative diffusion models, data points undergo a sequence of matrix transformations:

$$\mathbf{x}_{t+1} = A_t \mathbf{x}_t + \mathbf{b}_t, \quad t = 0, \ldots, T$$

- $\mathbf{x}_0$: 순수 노이즈 (초기 입력)
- $\mathbf{x}_T$: 최종 생성 이미지
- $A_t, \mathbf{b}_t$: $\mathbf{x}_t$와 $t$의 **비선형 함수** (신경망이 학습)

#### Whitening과의 연결

| 방향 | 과정 | Whitening 관점 |
|------|------|----------------|
| Forward | 이미지 → 노이즈 | 데이터를 점진적으로 whitening |
| Reverse | 노이즈 → 이미지 | whitened → 원래 구조 복원 |

---

### 1.4.2 Transformer 메커니즘 (상세)

#### 책 원문

> The transformer architecture, introduced in 2017 by Vaswani et al. in "Attention is All You Need", revolutionized AI by leveraging self-attention mechanisms.

#### 입력 표현

입력 시퀀스: $X = \{t_1, t_2, \ldots, t_n\} \in \mathbb{R}^{n \times d}$
- $n$: 시퀀스 길이
- $d$: 임베딩 차원
- $t_i \in \mathbb{R}^d$: i번째 토큰

#### Attention vs Self-Attention: 뭐가 다른가?

**먼저 알아야 할 것**: Attention은 2017년에 갑자기 나온 게 아닙니다!

```
Attention의 역사:

2014년: Seq2Seq Attention (Bahdanau et al.)
        - 번역: 영어 문장 → 프랑스어 문장
        - "영어 단어들 중 어디를 볼까?"

2017년: Self-Attention (Vaswani et al., "Attention is All You Need")
        - 같은 문장 안에서 "어디를 볼까?"
        - Transformer의 핵심 아이디어
```

**핵심 차이: Query가 어디서 오는가?**

| 종류 | Query 출처 | Key/Value 출처 | 예시 |
|------|-----------|---------------|------|
| **Attention** (2014) | 다른 곳 (디코더) | 다른 곳 (인코더) | 번역: "Je"가 "I", "am"을 봄 |
| **Self-Attention** (2017) | 같은 시퀀스 | 같은 시퀀스 | "cat"이 같은 문장의 "it"을 봄 |
| **Cross-Attention** | 한 곳 | 다른 곳 | 이미지 캡셔닝: 단어가 이미지를 봄 |

**비유로 이해하기**:

```
Attention (2014) = "통역사"
  영어 문장: [I] [love] [you]
  프랑스어 번역 중 "t'aime"를 쓸 때:
  → "영어 문장 중 어떤 단어가 관련 있나?" (love!)

  Query = 지금 쓰려는 프랑스어 단어
  Key/Value = 영어 문장

Self-Attention (2017) = "독서 토론"
  문장: [The] [cat] [sat] [because] [it] [was] [tired]
  "it"을 처리할 때:
  → "같은 문장 안에서 뭘 참고할까?" (cat!)

  Query = "it"
  Key/Value = 같은 문장의 모든 단어들
```

**결론**: All attention is NOT self-attention!
- Self-Attention은 Attention의 **특수한 경우**
- 2017 논문이 Self-Attention을 "처음 발명"한 건 아님
- 하지만 **"Self-Attention만으로 전체 모델을 만들자"**는 아이디어가 혁명적!

---

#### Q, K, V: 뭐가 주어지고, 뭐가 계산되나?

**헷갈리는 부분을 명확히!**

```
입력 (주어지는 것):
  X = 토큰 임베딩 시퀀스 (n개 단어, 각각 d차원)
      [[word1 벡터],
       [word2 벡터],
       [word3 벡터],
       ...
       [wordn 벡터]]   ← n×d 행렬

학습하는 것 (파라미터):
  W_Q, W_K, W_V ← 각각 d×d 행렬 (학습됨!)

계산되는 것:
  Q = X @ W_Q  (Query: "내가 뭘 찾고 싶은지")
  K = X @ W_K  (Key: "나는 어떤 정보를 가졌는지")
  V = X @ W_V  (Value: "내가 실제로 전달할 정보")
```

**비유: 도서관에서 책 찾기**

```
당신 (Query):
  "딥러닝 관련 책 찾고 있어요"

책들 (Key):
  책1: "이건 요리책이에요"
  책2: "이건 딥러닝 입문서예요"  ← 매칭!
  책3: "이건 역사책이에요"

매칭 후 가져오는 것 (Value):
  책2의 실제 내용

Q·K = "내 요청"과 "책 설명"의 유사도
V = 유사한 책에서 가져오는 실제 정보
```

**왜 Q, K, V를 분리하는가?**

```
분리하지 않으면?
  → 모든 단어가 같은 역할 (찾는 것 = 찾아지는 것 = 전달하는 것)

분리하면?
  → 유연성 증가!

예: "it"이 "cat"을 찾을 때
  - it의 Query: "나는 대명사야, 앞에 나온 명사 찾아"
  - cat의 Key: "나는 명사야, 대명사가 나를 참조할 수 있어"
  - cat의 Value: "고양이라는 의미 정보"
```

---

#### 잠깐, 아무도 역할을 가르쳐주지 않았는데?

**의문**: "W_Q는 찾는 역할, W_K는 찾아지는 역할"이라고 누가 정해준 적 없는데, 어떻게 다르게 학습되나?

**답: 계산 구조가 역할 분화를 강제한다!**

```
Attention 계산 흐름:

X ──┬── ×W_Q ──→ Q ──┐
    │                 ├── Q·Kᵀ ──→ softmax ──→ α
    ├── ×W_K ──→ K ──┘                          │
    │                                            │
    └── ×W_V ──→ V ────────────────────────────→ × ──→ output
                                               (α·V)
```

**역전파할 때 gradient가 다르게 흐른다!**

```
output = softmax(Q·Kᵀ/√d) · V = α · V

각 파라미터로 가는 gradient 경로:

∂L/∂W_V: Loss → output → α·V → V → W_V
         V는 "출력에 직접 연결"
         → W_V는 "좋은 정보 전달"하도록 학습

∂L/∂W_Q: Loss → output → α → softmax → Q·Kᵀ → Q → W_Q
         Q는 "매칭 점수 계산의 왼쪽"
         → W_Q는 "잘 찾도록" 학습

∂L/∂W_K: Loss → output → α → softmax → Q·Kᵀ → K → W_K
         K는 "매칭 점수 계산의 오른쪽"
         → W_K는 "잘 찾아지도록" 학습
```

**수식으로 보면**:

$\text{score} = Q \cdot K^\top$ 에서:

$$\frac{\partial \text{score}}{\partial Q} = K \quad \leftarrow \text{Q의 gradient는 K 값에 의존}$$

$$\frac{\partial \text{score}}{\partial K} = Q \quad \leftarrow \text{K의 gradient는 Q 값에 의존}$$

서로 다른 값으로 곱해지니까, $W_Q$와 $W_K$의 업데이트 방향이 다름!

**비유: 악수**

```
Q·Kᵀ = 두 사람이 악수하는 것

왼손(Q): "상대방 오른손(K)을 찾아서 잡기"
오른손(K): "상대방 왼손(Q)에게 잡히기"

같은 "악수" 동작이지만,
왼손과 오른손이 하는 역할은 다름!

gradient도 마찬가지:
- W_Q는 "K를 잘 찾는 방향"으로 업데이트
- W_K는 "Q에게 잘 찾아지는 방향"으로 업데이트
- W_V는 "좋은 정보를 전달하는 방향"으로 업데이트
```

**결론**:

```
"Q는 찾는 역할을 해라"라고 가르친 적 없음!

하지만 계산 구조상:
- Q는 Q·Kᵀ의 왼쪽에 있고
- K는 Q·Kᵀ의 오른쪽에 있고
- V는 출력으로 직접 가고

→ gradient가 다른 경로로 흐름
→ W_Q, W_K, W_V가 다른 방향으로 업데이트됨
→ 결과적으로 다른 "역할"을 학습하게 됨!
```

---

#### Gradient 업데이트 수식의 각 항 설명

위에서 `W_K -= lr × (... × Q)` 같은 표기를 썼는데, 각 항의 의미:

**`lr` = Learning Rate (학습률)**

```
lr = 0.001  (예시)

역할: gradient 방향으로 얼마나 크게 이동할지 결정하는 하이퍼파라미터
- 작을수록: 천천히, 안정적으로 학습
- 클수록: 빠르지만 불안정할 수 있음
```

**`...` = 생략된 Chain Rule 항들**

Loss에서 score까지 역전파되는 중간 gradient들:

**전체 chain rule**:

$$\frac{\partial L}{\partial W_K} = \underbrace{\frac{\partial L}{\partial \text{output}} \times \frac{\partial \text{output}}{\partial \alpha} \times \frac{\partial \alpha}{\partial \text{score}}}_{\text{이 부분이 "..."}} \times \frac{\partial \text{score}}{\partial K} \times \frac{\partial K}{\partial W_K}$$

**구체적으로 펼치면**:
- $\text{score} = Q \cdot K^\top / \sqrt{d}$
- $\alpha = \text{softmax}(\text{score})$ ← $\frac{\partial \alpha}{\partial \text{score}} = \alpha(1-\alpha)$ 형태
- $\text{output} = \alpha \cdot V$
- $L = f(\text{output}, \text{target})$ ← $\frac{\partial L}{\partial \text{output}} = (\text{예측} - \text{정답})$ 형태

**왜 `... × Q` 형태가 되나?**

$\frac{\partial \text{score}}{\partial K} = Q$ ← $\text{score} = Q \cdot K^\top$ 니까

$$\frac{\partial L}{\partial K} = \left(\frac{\partial L}{\partial \text{output}} \times \frac{\partial \text{output}}{\partial \alpha} \times \frac{\partial \alpha}{\partial \text{score}}\right) \times Q = (\ldots) \times Q$$

→ $Q$가 gradient의 "방향"을 결정한다!

**실제 코드에서의 형태**:

```python
# PyTorch에서는 자동 미분이 처리하지만, 풀어쓰면:
dK = upstream_gradient @ Q  # K의 gradient는 Q에 의존
W_K = W_K - lr * (X.T @ dK)
```

**핵심 포인트**:
- `lr`: 사람이 정하는 상수 (학습 속도)
- `...`: softmax, loss를 거쳐 자동 계산되는 gradient 항들
- 중요한 건 **어떤 값이 곱해지는가** → 그것이 업데이트 방향을 결정

---

#### Self-Attention (책 원문 수식)

**Step 1**: Query, Key, Value 계산
$$Q = \hat{t}_{n+1}^{(k)} W_Q, \quad K = XW_K, \quad V = XW_V$$

**Step 2**: Attention 가중치
$$\alpha_i^{(k)} = \text{softmax}\left(\frac{Q \cdot K^\top}{\sqrt{d}}\right)_i = \frac{\exp\left(\frac{Q K_{i,:}^\top}{\sqrt{d}}\right)}{\sum_{j=1}^n \exp\left(\frac{Q K_{j,:}^\top}{\sqrt{d}}\right)}$$

**Step 3**: 가중 합산
$$\mathbf{z}^{(k)} = \sum_{i=1}^n \alpha_i^{(k)} V_i$$

#### 왜 $\sqrt{d}$로 나누는가?

**문제 상황**:

내적: $q \cdot k = q_1 k_1 + q_2 k_2 + \cdots + q_d k_d$

$q, k$가 평균 0, 분산 1인 랜덤 벡터라고 가정하면:
- 각 항 $q_i k_i$의 분산 $\approx 1$
- $d$개 항의 합의 분산 $\approx d$ (독립이므로 합산)

$d = 512$면?
- 내적값이 $-30 \sim +30$ 범위로 커짐
- $\text{softmax}(30) \approx 1.0$, $\text{softmax}(-30) \approx 0.0$
- → 거의 one-hot! (하나만 1, 나머지 0)
- → gradient가 거의 0 (학습 안 됨!)

**해결: $\sqrt{d}$로 나누기**

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d}}\right) = \frac{d}{d} = 1$$

이제 softmax 입력이 적당한 범위 $(-3 \sim +3)$
→ 부드러운 확률 분포 → gradient가 잘 흐름 → 학습 가능!

**시각화**:

| $\sqrt{d}$ 없이 | $\sqrt{d}$로 나누면 |
|----------------|-------------------|
| $\text{softmax}([5, 30, -20])$ | $\text{softmax}([0.5, 1.2, -0.8])$ |
| $= [0.00, 1.00, 0.00]$ | $= [0.25, 0.50, 0.25]$ |
| ↑ 극단적! | ↑ 부드러움! |

#### 비선형 변환 + 정규화 (책 원문)

$$\hat{t}_{n+1}^{(k+1)} = \text{LayerNorm}\left(\sigma\left(W_2 \sigma(W_1 \mathbf{z}^{(k)}) + \mathbf{b}\right)\right)$$

**Layer Normalization**:
$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \bm{\gamma} + \bm{\beta}$$

- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$: 평균
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$: 분산
- $\bm{\gamma}, \bm{\beta}$: 학습 가능한 파라미터

---

#### Layer Norm vs Batch Norm: 왜 Transformer는 LayerNorm을 쓰나?

**먼저, 왜 정규화가 필요한가?**

신경망이 깊어지면 생기는 문제:

| 층 | 출력 분포 | 문제 |
|----|----------|------|
| 층 1 | 평균 0, 분산 1 | 정상 |
| 층 10 | 평균 5, 분산 100 | 스케일 변함 |
| 층 50 | 평균 ???, 분산 ??? | 폭발 또는 소멸 |

→ 학습 불안정, gradient vanishing/exploding
→ 해결책: 각 층의 출력을 정규화!

**Batch Normalization (2015)** - CNN의 표준

데이터 형태: $(\text{Batch}, \text{Feature})$

| | Feature 1 | Feature 2 | Feature 3 |
|---|---|---|---|
| img1 | $x_{11}$ | $x_{12}$ | $x_{13}$ |
| img2 | $x_{21}$ | $x_{22}$ | $x_{23}$ |
| img3 | $x_{31}$ | $x_{32}$ | $x_{33}$ |
| img4 | $x_{41}$ | $x_{42}$ | $x_{43}$ |
| | ↓ | ↓ | ↓ |
| | $\mu_1, \sigma_1^2$ | $\mu_2, \sigma_2^2$ | $\mu_3, \sigma_3^2$ |

**"같은 feature에 대해 배치 내 모든 샘플의 평균/분산"** (세로)

**Layer Normalization (2016)** - Transformer의 표준

데이터 형태: $(\text{Batch}, \text{Feature})$

| | dim1 | dim2 | dim3 | 정규화 |
|---|---|---|---|---|
| word1 | $x_{11}$ | $x_{12}$ | $x_{13}$ | ← $\mu_1, \sigma_1^2$ |
| word2 | $x_{21}$ | $x_{22}$ | $x_{23}$ | ← $\mu_2, \sigma_2^2$ |
| word3 | $x_{31}$ | $x_{32}$ | $x_{33}$ | ← $\mu_3, \sigma_3^2$ |
| word4 | $x_{41}$ | $x_{42}$ | $x_{43}$ | ← $\mu_4, \sigma_4^2$ |

**"각 샘플 내에서 모든 feature의 평균/분산"** (가로)

**시각적 비교**:

입력 텐서: $(\text{Batch}=4, \text{Features}=3)$

| | Feature 1 | Feature 2 | Feature 3 |
|---|---|---|---|
| Sample 1 | 0.5 | 1.2 | -0.3 |
| Sample 2 | 0.8 | 0.9 | 0.1 |
| Sample 3 | -0.2 | 1.5 | 0.7 |
| Sample 4 | 0.3 | 0.6 | -0.1 |

- **BatchNorm**: ↓ 세로로 정규화 (같은 feature, 다른 샘플)
  - Feature 1의 $\mu, \sigma^2$ 계산: $(0.5, 0.8, -0.2, 0.3)$

- **LayerNorm**: → 가로로 정규화 (같은 샘플, 다른 feature)
  - Sample 1의 $\mu, \sigma^2$ 계산: $(0.5, 1.2, -0.3)$

**왜 Transformer는 LayerNorm?**

| 문제 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| **가변 시퀀스 길이** | ❌ 문제 | ✅ OK |
| **작은 배치 크기** | ❌ 통계 불안정 | ✅ OK |
| **추론 시 단일 샘플** | ❌ running mean 필요 | ✅ 그대로 사용 |
| **시퀀스 내 위치 무관** | ❌ 위치별로 다름 | ✅ 동일하게 처리 |

**상세 설명**:

**문제 1: 가변 시퀀스 길이**
- **BatchNorm**: 배치 내 문장들 "I love you" (3단어), "Hello" (1단어), "How are you doing" (4단어)
  - 같은 "위치"끼리 평균? 위치 3에는 일부 문장만 있음! → padding 처리 복잡
- **LayerNorm**: 각 토큰을 독립적으로 정규화 → 시퀀스 길이 상관없음!

**문제 2: 배치 크기**
- **BatchNorm**: 배치 크기 $= 2$면? → 2개 샘플로 $\mu, \sigma^2$ 추정 = 매우 불안정
- **LayerNorm**: feature 차원이 512면? → 512개 값으로 $\mu, \sigma^2$ 추정 = 충분히 안정

**문제 3: 추론 시 (inference)**
- **BatchNorm**: 학습 시 배치 통계, 추론 시 running mean/var 사용 → 학습과 추론이 다르게 동작
- **LayerNorm**: 학습 = 추론 (동일하게 동작) → 더 예측 가능하고 안정적

**한 줄 요약**:
- **BatchNorm**: "다른 샘플들과 비교해서 정규화" → 배치가 커야 하고, 고정 길이가 좋음 (CNN)
- **LayerNorm**: "자기 자신 내에서 정규화" → 배치/길이 상관없음 (Transformer, RNN)

**코드로 비교**:

```python
import torch
import torch.nn as nn

x = torch.randn(4, 512)  # batch=4, features=512

# BatchNorm: feature 차원(512) 지정
bn = nn.BatchNorm1d(512)
out_bn = bn(x)  # 각 feature별로 batch 평균/분산

# LayerNorm: feature 차원(512) 지정
ln = nn.LayerNorm(512)
out_ln = ln(x)  # 각 샘플별로 feature 평균/분산

# 확인
print(out_bn[0].mean(), out_bn[0].std())  # 0이 아님 (샘플별 아님)
print(out_ln[0].mean(), out_ln[0].std())  # ≈ 0, ≈ 1 (샘플 내 정규화)
```

#### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

각 head가 서로 다른 관계 패턴 학습:
- head 1: 문법적 관계 (주어-동사)
- head 2: 의미적 관계 (고양이-귀엽다)
- head 3: 지시어 관계 ("it" → "cat")

#### 최종 토큰 예측

$$p(t_{n+1}|X) = \text{softmax}(W_{\text{out}} \hat{t}_{n+1}(X) + \mathbf{b}_{\text{out}})$$

---

## Exercises (연습문제)

### Exercise 1.1.1: Matrix Multiplication and Elliptical Dynamics

> **문제**: 초기 벡터 $\mathbf{x}_0 = (a, 0)^\top$에 행렬 $A$를 반복 적용하여 타원 위의 점들을 생성하는 $A$ 설계하기.

**(a)** 반장축 비율 $a/b$인 타원 위에 $\mathbf{x}_t = A^t \mathbf{x}_0$ 놓기

**풀이**:
회전 행렬을 스케일링과 결합:
$$A = \begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix} \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} 1/a & 0 \\ 0 & 1/b \end{pmatrix}$$

**(b)** $t \to \infty$일 때 타원 전체를 덮으려면?

**답**: $\theta$가 $\pi$의 **무리수 배**여야 함
- 유리수 배: 주기적 → 유한 개 점
- 무리수 배: 비주기적 → 조밀하게 채움

---

### Exercise 1.1.3: Einstein Summation (KAIST Challenge 1)

**(1)** $\mathbf{y} = A\mathbf{x}$:
$$y_i = A_{ij} x_j$$

**(2)** $\|A\|_F^2$:
$$\|A\|_F^2 = A_{ij} A_{ij}$$

**NumPy 구현**:
```python
import numpy as np

# (1) 행렬-벡터 곱
y = np.einsum('ij,j->i', A, x)

# (2) Frobenius 노름
frob_sq = np.einsum('ij,ij->', A, A)
```

---

### Exercise 1.2.2: SVD for Matrix Completion (KAIST Challenge 2)

> **문제**: 90% 에너지를 유지하는 최소 k 찾기

**Python 코드**:
```python
import numpy as np

def find_k_for_energy(A, threshold=0.90):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    total_energy = np.sum(s**2)
    cumulative = np.cumsum(s**2) / total_energy

    k = np.searchsorted(cumulative, threshold) + 1
    return k, cumulative

# 예시
A = np.random.randn(100, 50)
k, cumsum = find_k_for_energy(A)
print(f"90% 에너지 유지: k = {k}")
```

---

## Data Whitening - VAE/Diffusion 연결

### 왜 중요한가?

SVD는 **찌그러진 타원 데이터를 동그란 공(표준정규분포)으로 만드는** 도구

### 수식

공분산 행렬:
$$C = \frac{1}{n}X^\top X = V\Sigma^2 V^\top$$

Whitening 변환:
$$X_{\text{white}} = X V \Sigma^{-1}$$

### 생성 AI에서의 역할

| 모델 | Whitening 역할 |
|------|----------------|
| **VAE** | KL Divergence가 잠재 공간을 $\mathcal{N}(0,I)$로 강제 |
| **Diffusion** | Forward가 데이터를 순수 노이즈로 녹임 |
| **Normalizing Flow** | 복잡한 분포를 가우시안으로 변환 |

---

## 노트북 가이드

### [MNIST-SVD.ipynb](LinearAlgebra/MNIST-SVD.ipynb)

**배우는 것**:
- 이미지 배치를 행렬로 변환 (100×784)
- 평균 이미지와 "eigendigit" 시각화
- Rank-k 재구성으로 압축 효과 확인

**핵심 코드**:
```python
X_centered = X - mean_image
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Rank-k 재구성
x_reconstructed = mean_image + (x @ Vt[:k].T) @ Vt[:k]
```

### [Ex1-1-1image.ipynb](LinearAlgebra/Ex1-1-1image.ipynb)

**배우는 것**:
- RGB 이미지: $T \in \mathbb{R}^{H \times W \times 3}$
- 채널 추출, 평균 강도 계산

---

## 다음 장과의 연결

| 연결 | 챕터 | 개념 |
|------|------|------|
| → Ch.2 | Calculus | Jacobian, Chain Rule (KAIST Ch.3) |
| → Ch.3 | Optimization | Gradient Descent, Adam (KAIST Ch.5-6) |
| → Ch.4 | Neural Networks | CNN, ResNet (KAIST Ch.7-8) |
| → Ch.5 | Probability | 공분산, 가우시안 분포 |
| → Ch.6 | Information | KL Divergence, VAE |
| → Ch.7 | Stochastic | Diffusion, Brownian Motion |

---

## 핵심 요약

| 개념 | 정의 | AI 응용 |
|------|------|---------|
| **Vector** | 순서 있는 숫자 모음 | 데이터 표현 |
| **Matrix** | 선형 변환의 표현 | 가중치, 회전, 투영 |
| **Tensor** | 다차원 배열 | 이미지, 비디오, 배치 |
| **Einstein** | 반복 인덱스 = 합산 | GPU 텐서 연산 |
| **Convolution** | 슬라이딩 내적 | CNN 특징 추출 |
| **SVD** | $X = U\Sigma V^\top$ | 압축, PCA, LoRA |
| **Whitening** | 타원 → 원 | VAE, Diffusion 전처리 |
| **Attention** | 가중 평균 | Transformer |

---

## 참고 자료

- **책 원본**: MathGenAIBook12_14_25.tex (Chapter 1)
- **KAIST Challenge 풀이**: [KAIST-challenges-solutions.md](KAIST-challenges-solutions.md)
- **Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/
- **원논문**: Vaswani et al., "Attention is All You Need" (2017)
