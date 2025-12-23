# Mathematics of Generative AI: Complete Study Guide

> **Target Audience**: Engineers wanting practical understanding of generative AI mathematics
> **Duration**: 10-12 weeks (flexible pacing)
> **Prerequisites**: Basic calculus, linear algebra, Python/PyTorch familiarity

---

## How to Use This Guide

This guide is designed for **practical understanding** - you'll learn the math by seeing it work in code. Each chapter follows this pattern:

1. **Why It Matters** - Real-world motivation
2. **Core Concepts** - What you need to know (no fluff)
3. **Hands-On Labs** - Notebooks to run and modify
4. **Engineering Intuition** - How to think about it
5. **Checkpoint** - Verify your understanding

---

## The Big Picture: What Are We Building?

```
YOUR GOAL: Understand how AI generates images, text, and more

The Math You Need:
┌─────────────────────────────────────────────────────────────┐
│  FOUNDATIONS          PROBABILISTIC         GENERTIC MODELS │
│  ───────────          ─────────────         ────────────── │
│  Ch.1 Linear Algebra  Ch.5 Probability      Ch.8 VAE/EBM   │
│  Ch.2 Calculus/ODE    Ch.6 Information      Ch.9 Diffusion │
│  Ch.3 Optimization    Ch.7 Stochastic       (GAN, Flow)    │
│  Ch.4 Neural Nets                                          │
└─────────────────────────────────────────────────────────────┘

Everything leads to: Score-Based Diffusion Models (Ch.9)
```

---

## Week-by-Week Schedule

| Week | Chapter | Focus | Time |
|------|---------|-------|------|
| 1 | Ch.1 | SVD, Whitening, Attention basics | 6-8 hrs |
| 2 | Ch.2 | Autodiff, ODEs, ResNet connection | 6-8 hrs |
| 3 | Ch.3 | SGD, Adam, Regularization | 5-6 hrs |
| 4 | Ch.4 | CNN → ResNet → Neural ODE | 8-10 hrs |
| 5 | Ch.5 | Distributions, Normalizing Flows | 6-8 hrs |
| 6 | Ch.6 | Entropy, KL Divergence, ELBO | 8-10 hrs |
| 7-8 | Ch.7 | MCMC, Langevin, Brownian Motion | 10-12 hrs |
| 9 | Ch.8 | Energy Models, VAE, GNN | 8-10 hrs |
| 10-12 | Ch.9 | Score-Based Diffusion (Capstone) | 12-15 hrs |

---

# Chapter 1: Linear Algebra (of AI)

## Why Engineers Care

Every neural network is just matrix multiplications + nonlinearities. Understanding matrices = understanding what your model actually does.

**Real applications:**
- **LoRA fine-tuning**: Uses low-rank SVD approximation
- **Attention mechanism**: Matrix operations between Q, K, V
- **Data preprocessing**: Whitening for stable training

## Core Concepts (What You Need to Know)

### 1.1 SVD: The Swiss Army Knife

```python
# Any matrix can be decomposed as:
X = U @ Sigma @ V.T

# U: How samples relate to hidden features
# Sigma: Importance of each feature (singular values)
# V: How original features combine into hidden features
```

**Engineering Intuition:**
- SVD = finding the "most important directions" in your data
- Keep top-k singular values = lossy compression
- 90% energy with 10% of dimensions = massive savings

**Lab**: Run `notebook/chapter1/LinearAlgebra/MNIST-SVD.ipynb`
- Modify k values (10, 50, 100, 200)
- Watch reconstruction quality vs compression tradeoff

### 1.2 Whitening: Preparing Data for Generation

```python
# Whitening transforms data so:
# - Mean = 0
# - Covariance = Identity matrix

X_white = (X - mean) @ V @ inv(Sigma)

# Result: Data looks like a "unit sphere" - no direction is special
```

**Why it matters for generative AI:**
- VAE forces latent space to be "whitened" (N(0,I))
- Diffusion adds noise until data becomes pure Gaussian (whitened)
- Easier to sample from uniform distributions

### 1.3 Attention: Dynamic Matrix Operations

```python
# Self-attention in one line:
output = softmax(Q @ K.T / sqrt(d)) @ V

# Q: "What am I looking for?"
# K: "What do I have to offer?"
# V: "What's my actual content?"
```

**The sqrt(d) trick:**
- Without it: softmax becomes one-hot (gradient dies)
- With it: softmax stays smooth (gradient flows)

## Chapter 1 Checkpoint

Can you answer these?
- [ ] What does each matrix in SVD represent?
- [ ] Why do we divide attention by sqrt(d)?
- [ ] How does whitening relate to VAE's latent space?

---

# Chapter 2: Calculus and Differential Equations

## Why Engineers Care

Backpropagation IS calculus. Neural ODEs ARE differential equations. If you understand this chapter, you understand how deep learning actually works.

## Core Concepts

### 2.1 Automatic Differentiation

```python
# PyTorch does this for you:
loss.backward()  # Computes ALL gradients automatically

# Two modes:
# Forward mode: Good for few inputs, many outputs
# Reverse mode: Good for many inputs, few outputs (NEURAL NETS!)
```

**Engineering fact:** Reverse mode AD is why deep learning works. Computing gradients costs ~3x forward pass, regardless of parameter count.

### 2.2 The ResNet-ODE Connection

```python
# ResNet block:
h_next = h + f(h)  # Skip connection!

# This is Euler's method for:
dh/dt = f(h)

# ResNet = Discretized ODE with dt=1
```

**Why this matters:**
- Explains why skip connections work (stable ODE integration)
- Leads to Neural ODE (continuous depth, adaptive compute)
- Memory efficient: O(1) vs O(depth)

### 2.3 Neural ODE

```python
# Instead of fixed layers:
h1 = layer1(h0)
h2 = layer2(h1)
...

# Solve an ODE:
h_T = odesolve(f_theta, h_0, t=[0, T])

# Depth becomes continuous!
```

**Lab**: Run `notebook/chapter4/NN/NeuralODE-Spiral.ipynb`
- Compare Neural ODE vs ResNet on spiral classification
- Visualize the continuous transformation

## Chapter 2 Checkpoint

- [ ] Why is reverse-mode AD efficient for neural nets?
- [ ] How is ResNet related to Euler's method?
- [ ] What's the memory advantage of Neural ODE?

---

# Chapter 3: Optimization

## Why Engineers Care

Your model is only as good as your optimizer. Understanding optimization = understanding why your training succeeds or fails.

## Core Concepts

### 3.1 Gradient Descent Variants

```python
# Vanilla SGD:
w = w - lr * grad

# Momentum (Polyak, 1964):
v = momentum * v + grad
w = w - lr * v

# Adam (2014) - The standard:
m = beta1 * m + (1-beta1) * grad      # First moment
v = beta2 * v + (1-beta2) * grad**2   # Second moment
m_hat = m / (1 - beta1**t)            # Bias correction
v_hat = v / (1 - beta2**t)
w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

### 3.2 Learning Rate: The Most Important Hyperparameter

```
Too high: Training explodes (loss = NaN)
Too low: Training takes forever
Just right: Smooth convergence

Modern approach: Warmup + Cosine decay
```

### 3.3 Regularization: L1 vs L2

```python
# L2 (Weight Decay): Prefers small weights
loss = loss + lambda * sum(w**2)
# Effect: Smooth solutions, no zeros

# L1 (Lasso): Prefers sparse weights
loss = loss + lambda * sum(abs(w))
# Effect: Many weights become exactly zero
```

**Lab**: Run `notebook/chapter3/Optimization/adaptive.ipynb`
- Compare SGD, Adam, RMSProp
- See how Adam adapts to different gradient scales

## Chapter 3 Checkpoint

- [ ] Why does Adam have bias correction?
- [ ] When would you use L1 vs L2 regularization?
- [ ] What happens if learning rate is too high?

---

# Chapter 4: Neural Networks

## Why Engineers Care

This is where theory meets practice. Understanding architectures = building better models.

## Core Concepts

### 4.1 CNN: Exploiting Structure

```python
# Key ideas:
# 1. Local connectivity: Each neuron sees small patch
# 2. Weight sharing: Same filter everywhere
# 3. Translation invariance: Cat is cat, regardless of position

# Convolution as structured matrix multiply:
y = conv(x, kernel)  # Actually a Toeplitz matrix!
```

### 4.2 ResNet: Why Depth Works

```python
# The problem: Deep nets have vanishing gradients
# The solution: Skip connections

def resnet_block(x):
    return x + F(x)  # Gradient flows through "+"!

# Even if F(x) is bad, gradient still flows through x
```

**Engineering insight:** Skip connections are like "gradient highways" - they let gradients flow unchanged through many layers.

### 4.3 From ResNet to Neural ODE to Diffusion

```
ResNet → Neural ODE → Diffusion

h_{t+1} = h_t + f(h_t)     # Discrete
dh/dt = f(h)               # Continuous
dx = f(x,t)dt + g(t)dW     # Stochastic (Diffusion!)
```

**Lab sequence:**
1. `notebook/chapter4/NN/ResNet9-Spiral.ipynb` - ResNet basics
2. `notebook/chapter4/NN/NeuralODE-Spiral.ipynb` - Continuous version
3. `notebook/chapter7/Stochastic/Langevin-DoubleWell.ipynb` - Add noise!

## Chapter 4 Checkpoint

- [ ] Why do skip connections help gradient flow?
- [ ] How is convolution related to matrix multiplication?
- [ ] What's the progression from ResNet to Diffusion?

---

# Chapter 5: Probability and Statistics

## Why Engineers Care

Generative models ARE probability. You're learning to sample from p(data).

## Core Concepts

### 5.1 Sampling Methods

```python
# Inverse Transform Sampling:
# If you can invert the CDF, you can sample!
u = uniform(0, 1)
x = CDF_inverse(u)  # Exact sample from distribution!

# When you can't invert (most cases):
# Use MCMC, importance sampling, or learn a sampler (VAE/Diffusion)
```

### 5.2 Normalizing Flows

```python
# Idea: Transform simple distribution to complex one
# Key constraint: Transformation must be invertible!

z ~ N(0, I)           # Simple base distribution
x = f(z)              # Invertible transform
p(x) = p(z) * |det(df/dz)|^{-1}  # Change of variables!

# Training: Maximize log p(x) directly
```

**Why flows matter:**
- Exact likelihood computation (unlike VAE/GAN)
- Exact inference (unlike VAE)
- But: Expensive Jacobian computation

### 5.3 Multivariate Gaussian

```python
# The most important distribution in ML:
p(x) = N(mu, Sigma)

# Key property: Linear transforms of Gaussians are Gaussian
# This is why VAE/Diffusion assume Gaussian noise!
```

**Lab**: Run `notebook/chapter5/Statistics/Normalizing-Flow-1D.ipynb`
- See how simple distributions transform to complex ones
- Visualize the Jacobian's role

## Chapter 5 Checkpoint

- [ ] What makes normalizing flows different from VAE?
- [ ] Why do we always use Gaussian noise?
- [ ] How does inverse CDF sampling work?

---

# Chapter 6: Entropy and Information Theory

## Why Engineers Care

This chapter explains the loss functions. Cross-entropy loss, KL divergence in VAE, ELBO - all from information theory.

## Core Concepts

### 6.1 Entropy: Measuring Uncertainty

```python
# Shannon entropy:
H(X) = -sum(p(x) * log(p(x)))

# High entropy = high uncertainty = hard to predict
# Low entropy = low uncertainty = easy to predict

# Uniform distribution: Maximum entropy
# Delta distribution: Zero entropy
```

### 6.2 KL Divergence: Comparing Distributions

```python
# "How different is Q from P?"
KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

# Properties:
# - Always >= 0
# - = 0 only when P = Q
# - NOT symmetric! KL(P||Q) != KL(Q||P)
```

**The two KLs in practice:**
- `KL(data || model)`: Mode covering (model spreads out)
- `KL(model || data)`: Mode seeking (model focuses on peaks)

### 6.3 ELBO: The VAE Training Objective

```python
# We want: log p(x)  (intractable!)
# We optimize: ELBO (lower bound)

ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
       ─────────────   ───────────────────
       Reconstruction   Regularization
       (decode well)    (match prior N(0,I))

# Maximizing ELBO ≈ Maximizing log p(x)
```

**Lab**: Run `notebook/chapter6/Info/autoencoder-entropy.ipynb`
- See reconstruction vs KL tradeoff
- Modify beta weight and observe effects

## Chapter 6 Checkpoint

- [ ] What does high entropy mean for a distribution?
- [ ] Why is KL divergence not symmetric?
- [ ] What do the two terms in ELBO represent?

---

# Chapter 7: Stochastic Processes

## Why Engineers Care

**Diffusion models are stochastic processes.** This chapter is the direct foundation.

## Core Concepts

### 7.1 Brownian Motion

```python
# Random walk in continuous time:
dW ~ N(0, dt)

# Properties:
# - Continuous but nowhere differentiable
# - Independent increments
# - W(t) ~ N(0, t)
```

### 7.2 Stochastic Differential Equations (SDEs)

```python
# General form:
dx = f(x,t)dt + g(t)dW
    ─────────   ──────
    Drift       Diffusion (noise)

# Diffusion model forward process:
dx = -0.5*beta(t)*x*dt + sqrt(beta(t))*dW
# Gradually adds noise until x ~ N(0, I)
```

### 7.3 Langevin Dynamics: Sampling via SDEs

```python
# Want to sample from p(x) ∝ exp(-E(x))
# Solution: Run this SDE until convergence:

dx = -∇E(x)dt + sqrt(2/beta)dW
     ─────────   ──────────────
     Go downhill   Random exploration

# At equilibrium: Samples from p(x)!
```

**This is the key insight for diffusion:**
- Langevin samples from energy-based model
- Score function ∇log p(x) = -∇E(x)
- Diffusion learns the score function!

**Lab**: Run `notebook/chapter7/Stochastic/Langevin-DoubleWell.ipynb`
- Watch particles settle into energy minima
- See how noise helps escape local minima

## Chapter 7 Checkpoint

- [ ] What's the difference between drift and diffusion in an SDE?
- [ ] How does Langevin dynamics sample from a distribution?
- [ ] Why is this related to diffusion models?

---

# Chapter 8: Energy-Based Models

## Why Engineers Care

VAE, RBM, and even diffusion can be seen through the energy lens. Understanding energy = understanding generative models.

## Core Concepts

### 8.1 Energy Functions

```python
# Core idea: Low energy = high probability
p(x) = exp(-E(x)) / Z

# Z = partition function (usually intractable)
# We learn E(x) such that data has low energy
```

### 8.2 VAE: Variational Inference + Neural Nets

```python
# Encoder: x → q(z|x) = N(mu(x), sigma(x))
# Decoder: z → p(x|z)

# Training objective (ELBO):
loss = reconstruction_loss + beta * KL_loss

# The KL term keeps latent space "nice":
# - Close to N(0,I)
# - Can sample z ~ N(0,I) and decode to new x
```

### 8.3 ELBO Derivation (Understand This!)

```
log p(x) = log ∫ p(x,z) dz
         = log ∫ p(x,z) * q(z|x)/q(z|x) dz
         ≥ ∫ q(z|x) log [p(x,z)/q(z|x)] dz  # Jensen's inequality
         = E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = ELBO
```

**Lab**: Run `notebook/chapter8/VI_ELBO_Toy.ipynb`
- Visualize variational inference
- See how q(z|x) approximates true posterior

## Chapter 8 Checkpoint

- [ ] What does the partition function Z represent?
- [ ] Why can't we compute log p(x) directly for VAE?
- [ ] What happens if we increase beta in beta-VAE?

---

# Chapter 9: Score-Based Diffusion (Capstone)

## Why Engineers Care

**State-of-the-art generative models.** DALL-E, Stable Diffusion, Midjourney - all diffusion.

## Core Concepts

### 9.1 The Score Function

```python
# Score = gradient of log probability
s(x) = ∇_x log p(x)

# Points toward higher probability regions
# If we know the score, we can sample via Langevin!
```

### 9.2 Forward and Reverse Processes

```python
# FORWARD (known, fixed):
# Add noise gradually until data becomes pure noise
dx = f(x,t)dt + g(t)dW
# x(0) = data, x(T) ≈ N(0,I)

# REVERSE (learned):
# Remove noise gradually to generate data
dx = [f(x,t) - g(t)² ∇log p(x,t)] dt + g(t)dW̄
# x(T) ~ N(0,I), x(0) ≈ data

# The key: Learn ∇log p(x,t) with a neural network!
```

### 9.3 Score Matching Training

```python
# Training objective:
loss = E[||s_theta(x_t, t) - ∇log p(x_t|x_0)||²]

# Denoising score matching (practical form):
# ∇log p(x_t|x_0) = (x_0 - x_t) / sigma_t²
# "Score points toward clean data!"

loss = E[||s_theta(x_t, t) - (x_0 - x_t)/sigma_t²||²]
```

### 9.4 Unified View of Generative Models

```
All generative models through diffusion lens:

VAE:
    - One step forward (encode)
    - One step reverse (decode)
    - Learned noise schedule

GAN:
    - Zero noise (deterministic)
    - Single step z → x
    - Optimal transport map

Flow:
    - ODE instead of SDE (no noise)
    - Invertible by construction

Diffusion:
    - Many steps
    - Fixed forward, learned reverse
    - State-of-the-art quality
```

**Lab sequence for Chapter 9:**
1. `notebook/chapter9/02-SGM-with-SDE-9grid.ipynb` - Basic diffusion
2. `notebook/chapter9/ring_vae_latent_diffusion.ipynb` - VAE vs Diffusion
3. `notebook/chapter9/03-SchrBridge-1D.ipynb` - Bridge diffusion

## Chapter 9 Checkpoint

- [ ] What is the score function and why is it useful?
- [ ] How does denoising score matching work?
- [ ] How are VAE, GAN, and Diffusion related?

---

# Quick Reference: Key Equations

## The Essentials

| Concept | Equation | Where Used |
|---------|----------|------------|
| SVD | $X = U\Sigma V^\top$ | Data compression, LoRA |
| Softmax Attention | $\text{softmax}(QK^\top/\sqrt{d})V$ | Transformers |
| Cross-Entropy | $-\sum y_i \log \hat{y}_i$ | Classification |
| KL Divergence | $\sum P \log(P/Q)$ | VAE, Diffusion |
| ELBO | $\mathbb{E}[\log p(x|z)] - D_{KL}(q||p)$ | VAE training |
| Langevin | $dx = -\nabla E \cdot dt + \sqrt{2\beta^{-1}}dW$ | Sampling |
| Forward SDE | $dx = f(x,t)dt + g(t)dW$ | Diffusion forward |
| Reverse SDE | $dx = [f - g^2\nabla\log p]dt + g d\bar{W}$ | Diffusion reverse |
| Score | $s(x,t) = \nabla_x \log p(x,t)$ | Diffusion training |

---

# Appendix: Troubleshooting Common Confusions

## "Why divide by sqrt(d) in attention?"

Without it, dot products grow with dimension d. Large values → softmax becomes one-hot → gradients vanish.

## "What's the difference between KL(P||Q) and KL(Q||P)?"

- KL(data||model): Penalizes model for missing modes
- KL(model||data): Penalizes model for hallucinating

VAE uses KL(q||p) because we want q to be "contained" within p.

## "Why does diffusion need many steps?"

Each step only removes a little noise. Like slowly focusing a blurry image - doing it all at once would be too hard.

## "How is ELBO a lower bound?"

Jensen's inequality: log(E[X]) ≥ E[log(X)] for concave functions. The gap is the KL divergence between q(z|x) and true posterior p(z|x).

---

# Next Steps After This Guide

1. **Implement from scratch**: Build a simple diffusion model on MNIST
2. **Read papers**: Start with DDPM (Ho et al., 2020)
3. **Experiment**: Modify hyperparameters in notebooks
4. **Extend**: Try conditional generation, classifier guidance

---

*This guide accompanies the Mathematics of Generative AI Book. For deeper mathematical treatment, consult the PDF (MathGenAIBook12_14_25.pdf).*
