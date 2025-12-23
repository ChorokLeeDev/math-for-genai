# Problem Sets: Mathematics of Generative AI

> **Purpose**: Reinforce understanding through practice problems
> **Structure**: Conceptual questions + Coding exercises + Challenge problems
> **Difficulty**: * Easy, ** Medium, *** Hard

---

## Chapter 1: Linear Algebra

### Conceptual Problems

**1.1* SVD Basics**
Given a 100×50 matrix X with rank 30:
- (a) What are the dimensions of U, Σ, V?
- (b) How many non-zero singular values exist?
- (c) If you keep the top 10 singular values, what's the reconstructed matrix's rank?

<details>
<summary>Solution</summary>

(a) U: 100×100 (or 100×30 in reduced form), Σ: 100×50 (or 30×30), V: 50×50 (or 50×30)

(b) Exactly 30 (equals the rank)

(c) Rank 10 (you're projecting onto a 10-dimensional subspace)
</details>

---

**1.2** Energy Preservation**
A dataset has singular values [10, 5, 3, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05].
- (a) What's the total energy?
- (b) How many components for 90% energy?
- (c) How many for 99%?

<details>
<summary>Solution</summary>

(a) Total energy = Σσᵢ² = 100 + 25 + 9 + 4 + 1 + 0.25 + 0.09 + 0.04 + 0.01 + 0.0025 = 139.39

(b) Top 1: 100/139.39 = 71.7%, Top 2: 125/139.39 = 89.7%, Top 3: 134/139.39 = 96.1%
→ **3 components for 90%**

(c) Top 4: 138/139.39 = 99.0% → **4 components for 99%**
</details>

---

**1.3*** Whitening Properties**
After whitening transformation X_white = XV Σ⁻¹:
- (a) Prove that Cov(X_white) = I
- (b) Why is this useful for VAE?
- (c) What happens if a singular value is 0?

<details>
<summary>Solution</summary>

(a) Cov(X_white) = (X_white)ᵀX_white / n = (Σ⁻¹Vᵀ)(XᵀX/n)(VΣ⁻¹) = Σ⁻¹(Σ²)Σ⁻¹ = I

(b) VAE assumes p(z) = N(0,I). Pre-whitened data already matches this structure, making the encoder's job easier.

(c) Division by zero! This indicates a degenerate dimension (no variance). Solution: remove that dimension before whitening.
</details>

---

**1.4** Attention Scaling**
In attention, we divide by √d. If d=512 and q,k are unit Gaussians:
- (a) What's E[qᵀk] without scaling?
- (b) What's Var[qᵀk]?
- (c) What happens to softmax([0, 100, 0, 0])?

<details>
<summary>Solution</summary>

(a) E[qᵀk] = E[Σᵢ qᵢkᵢ] = Σᵢ E[qᵢ]E[kᵢ] = 0 (since E[qᵢ]=0)

(b) Var[qᵀk] = Σᵢ Var[qᵢkᵢ] = Σᵢ E[qᵢ²]E[kᵢ²] = d × 1 × 1 = 512

(c) softmax([0,100,0,0]) ≈ [0, 1, 0, 0] — nearly one-hot, gradient ≈ 0
</details>

---

### Coding Exercises

**1.5** Implement SVD from scratch**
```python
def my_svd(X, k=None):
    """
    Compute SVD of X using eigendecomposition of XᵀX.
    Returns U, S, Vt (top k components if specified)

    Hint:
    1. Compute XᵀX
    2. Eigendecompose to get V and λ
    3. S = sqrt(λ)
    4. U = XV/S
    """
    # Your code here
    pass

# Test
X = np.random.randn(100, 50)
U, S, Vt = my_svd(X)
X_reconstructed = U @ np.diag(S) @ Vt
assert np.allclose(X, X_reconstructed)
```

<details>
<summary>Solution</summary>

```python
def my_svd(X, k=None):
    XtX = X.T @ X
    eigenvalues, V = np.linalg.eigh(XtX)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Handle numerical issues
    eigenvalues = np.maximum(eigenvalues, 0)
    S = np.sqrt(eigenvalues)

    # Compute U
    U = X @ V / (S + 1e-10)

    if k is not None:
        U = U[:, :k]
        S = S[:k]
        V = V[:, :k]

    return U, S, V.T
```
</details>

---

**1.6** Attention implementation**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implement scaled dot-product attention.
    Q, K, V: (batch, seq_len, d_k)
    Returns: attention output and weights
    """
    # Your code here
    pass

# Test
B, L, D = 2, 10, 64
Q = np.random.randn(B, L, D)
K = np.random.randn(B, L, D)
V = np.random.randn(B, L, D)
output, weights = scaled_dot_product_attention(Q, K, V)
assert output.shape == (B, L, D)
assert weights.shape == (B, L, L)
```

---

## Chapter 2: Calculus and ODEs

### Conceptual Problems

**2.1* Chain Rule**
For f(x) = σ(Wx + b) where σ is sigmoid:
- (a) Compute ∂f/∂W
- (b) Compute ∂f/∂b
- (c) Why does sigmoid cause gradient problems?

<details>
<summary>Solution</summary>

(a) ∂f/∂W = σ'(Wx+b) ⊗ x = σ(Wx+b)(1-σ(Wx+b)) ⊗ x

(b) ∂f/∂b = σ'(Wx+b) = σ(Wx+b)(1-σ(Wx+b))

(c) σ'(x) peaks at 0.25 and goes to 0 for large |x|. In deep networks, multiplying many small gradients → vanishing gradient.
</details>

---

**2.2** Forward vs Reverse AD**
For f: ℝⁿ → ℝᵐ:
- (a) Forward mode: How many passes to get full Jacobian?
- (b) Reverse mode: How many passes?
- (c) For neural net with 10M params and scalar loss, which is better?

<details>
<summary>Solution</summary>

(a) Forward: n passes (one per input dimension)

(b) Reverse: m passes (one per output dimension)

(c) n = 10M, m = 1. Reverse mode needs 1 pass vs 10M. **Reverse is dramatically better!**
</details>

---

**2.3** ResNet as ODE**
Given ResNet block h_{t+1} = h_t + f(h_t):
- (a) What ODE does this discretize?
- (b) What's the "step size"?
- (c) If f(h) = -h, what happens as t → ∞?

<details>
<summary>Solution</summary>

(a) dh/dt = f(h) (first-order ODE)

(b) Step size dt = 1 (Euler method)

(c) Solution is h(t) = h(0)e^{-t} → 0. The system is stable, decaying exponentially.
</details>

---

### Coding Exercises

**2.4** Euler's Method**
```python
def euler_solve(f, h0, t_span, dt):
    """
    Solve dh/dt = f(h, t) using Euler's method.

    Args:
        f: function(h, t) -> dh/dt
        h0: initial condition
        t_span: (t_start, t_end)
        dt: time step

    Returns:
        times, solutions
    """
    # Your code here
    pass

# Test: Solve dh/dt = -h, h(0) = 1
# Analytical solution: h(t) = e^{-t}
f = lambda h, t: -h
times, solutions = euler_solve(f, 1.0, (0, 5), 0.01)
assert np.abs(solutions[-1] - np.exp(-5)) < 0.01
```

---

**2.5*** Neural ODE Layer**
```python
class NeuralODELayer(nn.Module):
    """
    Implement a simple Neural ODE layer.

    The forward pass should:
    1. Define f(h, t) as a small MLP
    2. Use torchdiffeq.odeint to integrate
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Your code here

    def forward(self, h0, t_span):
        # Your code here
        pass
```

---

## Chapter 3: Optimization

### Conceptual Problems

**3.1* Learning Rate**
- (a) LR too high: what happens to loss?
- (b) LR too low: what happens to training?
- (c) Why warmup + decay?

<details>
<summary>Solution</summary>

(a) Loss oscillates or diverges (NaN)

(b) Training is slow, may get stuck in local minima

(c) Warmup: Initial gradients are noisy, small LR prevents instability. Decay: As we approach minimum, smaller steps prevent overshooting.
</details>

---

**3.2** Adam vs SGD**
- (a) What are Adam's m and v tracking?
- (b) Why bias correction?
- (c) When might SGD beat Adam?

<details>
<summary>Solution</summary>

(a) m: first moment (running average of gradients), v: second moment (running average of squared gradients)

(b) At initialization, m=v=0. Without correction, initial estimates are biased toward 0.

(c) SGD often generalizes better on well-tuned problems (e.g., ImageNet ResNets). Adam can converge to sharper minima.
</details>

---

**3.3** L1 vs L2**
Weight vector w = [3, 0.01, -2, 0.001]:
- (a) Compute L1 norm
- (b) Compute L2 norm squared
- (c) Which regularizer drives 0.01 and 0.001 to exactly 0?

<details>
<summary>Solution</summary>

(a) L1 = |3| + |0.01| + |-2| + |0.001| = 5.011

(b) L2² = 9 + 0.0001 + 4 + 0.000001 = 13.0001

(c) **L1** (Lasso) drives small weights to exactly 0 due to the non-smooth point at origin. L2 only makes them small, never exactly 0.
</details>

---

### Coding Exercises

**3.4** Implement Adam**
```python
class MyAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data.numpy()) for p in self.params]
        self.v = [np.zeros_like(p.data.numpy()) for p in self.params]

    def step(self):
        """Perform one optimization step."""
        self.t += 1
        # Your code here: update m, v, and params
        pass

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

---

## Chapter 4: Neural Networks

### Conceptual Problems

**4.1* CNN Receptive Field**
A CNN with 3 layers of 3×3 convolutions (stride 1, no padding):
- (a) What's the receptive field of one output neuron?
- (b) Add a 2×2 max pool after layer 2. New receptive field?

<details>
<summary>Solution</summary>

(a) Layer 1: 3×3, Layer 2: 5×5, Layer 3: **7×7** (each layer adds 2 to each dimension)

(b) Pooling doubles the effective receptive field: **14×14**
</details>

---

**4.2** Skip Connections**
- (a) Why do skip connections help gradient flow?
- (b) What's the Jacobian of f(x) = x + g(x)?
- (c) Why is this better than f(x) = g(x)?

<details>
<summary>Solution</summary>

(a) The gradient of x+g(x) with respect to x is I + ∂g/∂x. Even if g saturates, gradient is at least I.

(b) Jacobian = I + Jg (identity + Jacobian of g)

(c) For f(x)=g(x), gradient must pass through g. If ||Jg|| < 1 for many layers, gradient vanishes. With skip, we always have the identity term.
</details>

---

**4.3*** Neural ODE Memory**
ResNet-100 vs Neural ODE with 100 evaluation points:
- (a) ResNet forward memory usage?
- (b) ResNet backward memory (naive)?
- (c) Neural ODE with adjoint method memory?

<details>
<summary>Solution</summary>

(a) Forward: O(100) = O(L) - must store all activations

(b) Backward: O(100) - need stored activations for gradients

(c) **O(1)** - adjoint method recomputes forward pass, only needs endpoints
</details>

---

### Coding Exercises

**4.4** ResNet Block**
```python
class ResidualBlock(nn.Module):
    """
    Implement a basic residual block:
    output = x + F(x)
    where F = Conv -> BN -> ReLU -> Conv -> BN
    """
    def __init__(self, channels):
        super().__init__()
        # Your code here

    def forward(self, x):
        # Your code here
        pass

# Test
block = ResidualBlock(64)
x = torch.randn(1, 64, 32, 32)
y = block(x)
assert y.shape == x.shape
```

---

## Chapter 5: Probability

### Conceptual Problems

**5.1* Sampling Methods**
- (a) When can you use inverse CDF sampling?
- (b) Why can't you use it for a neural network's output distribution?
- (c) What's the alternative?

<details>
<summary>Solution</summary>

(a) When you can analytically invert the CDF

(b) Neural network defines complex distribution - no analytical CDF inverse

(c) Reparameterization trick (VAE), MCMC, or learn a sampler (flow, diffusion)
</details>

---

**5.2** Change of Variables**
If z ~ N(0,1) and x = exp(z):
- (a) What distribution is x? (name it)
- (b) Write p(x) using change of variables
- (c) Why does the Jacobian appear?

<details>
<summary>Solution</summary>

(a) Log-normal distribution

(b) p(x) = p(z)|dz/dx| = (1/√(2π)) exp(-log²(x)/2) × (1/x)

(c) Probability mass must be conserved. Stretching/compressing space changes density inversely.
</details>

---

**5.3** Normalizing Flow**
A flow transforms z ~ N(0,1) through f₁, f₂, f₃ to get x.
- (a) Write log p(x)
- (b) Why must each fᵢ be invertible?
- (c) What's the computational cost concern?

<details>
<summary>Solution</summary>

(a) log p(x) = log p(z) - Σᵢ log|det Jfᵢ| where z = f₃⁻¹(f₂⁻¹(f₁⁻¹(x)))

(b) To compute z from x and to compute exact likelihood

(c) Computing det(J) is O(d³) for general d×d Jacobian. Need special architectures (coupling layers) for O(d).
</details>

---

## Chapter 6: Information Theory

### Conceptual Problems

**6.1* Entropy Calculations**
- (a) H(fair coin) = ?
- (b) H(biased coin, P(H)=0.9) = ?
- (c) H(always heads) = ?

<details>
<summary>Solution</summary>

(a) H = -0.5 log₂(0.5) - 0.5 log₂(0.5) = **1 bit**

(b) H = -0.9 log₂(0.9) - 0.1 log₂(0.1) ≈ **0.47 bits**

(c) H = -1 log₂(1) = **0 bits** (no uncertainty)
</details>

---

**6.2** KL Divergence**
P = [0.5, 0.5], Q = [0.9, 0.1]:
- (a) Compute KL(P||Q)
- (b) Compute KL(Q||P)
- (c) Why are they different?

<details>
<summary>Solution</summary>

(a) KL(P||Q) = 0.5 log(0.5/0.9) + 0.5 log(0.5/0.1) = 0.5(-0.85) + 0.5(2.32) = 0.74 bits

(b) KL(Q||P) = 0.9 log(0.9/0.5) + 0.1 log(0.1/0.5) = 0.9(0.85) + 0.1(-2.32) = 0.53 bits

(c) KL is not symmetric! KL(P||Q) penalizes Q having low probability where P is high.
</details>

---

**6.3*** ELBO Derivation**
Starting from log p(x), derive the ELBO:
- (a) Introduce q(z|x) via importance sampling
- (b) Apply Jensen's inequality
- (c) Identify the gap between log p(x) and ELBO

<details>
<summary>Solution</summary>

(a) log p(x) = log ∫ p(x,z) dz = log ∫ p(x,z) q(z|x)/q(z|x) dz = log E_q[p(x,z)/q(z|x)]

(b) ≥ E_q[log p(x,z)/q(z|x)] = E_q[log p(x|z) + log p(z) - log q(z|x)]
    = E_q[log p(x|z)] - KL(q(z|x)||p(z)) = ELBO

(c) Gap = log p(x) - ELBO = KL(q(z|x) || p(z|x)) (KL to true posterior)
</details>

---

### Coding Exercises

**6.4** Cross-Entropy Loss**
```python
def cross_entropy_loss(logits, targets):
    """
    Implement cross-entropy loss from scratch.

    Args:
        logits: (batch, num_classes) raw scores
        targets: (batch,) class indices

    Returns:
        scalar loss

    Steps:
    1. Compute log_softmax (numerically stable)
    2. Select correct class probabilities
    3. Take negative mean
    """
    # Your code here
    pass

# Test
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])
loss = cross_entropy_loss(logits, targets)
expected = F.cross_entropy(logits, targets)
assert torch.abs(loss - expected) < 1e-5
```

---

## Chapter 7: Stochastic Processes

### Conceptual Problems

**7.1* Brownian Motion**
For W_t (standard Brownian motion):
- (a) What's E[W_t]?
- (b) What's Var[W_t]?
- (c) What's Cov[W_s, W_t] for s < t?

<details>
<summary>Solution</summary>

(a) E[W_t] = 0

(b) Var[W_t] = t

(c) Cov[W_s, W_t] = min(s,t) = s (since W_t = W_s + (W_t - W_s), independent increments)
</details>

---

**7.2** Langevin Dynamics**
For the SDE: dx = -∇E(x)dt + √(2/β)dW
- (a) What distribution does it converge to?
- (b) What does β control?
- (c) What happens as β → ∞?

<details>
<summary>Solution</summary>

(a) Stationary distribution p(x) ∝ exp(-βE(x)) (Boltzmann distribution)

(b) β = inverse temperature. High β: peaked at minima. Low β: more uniform.

(c) β → ∞: distribution concentrates on global minimum of E(x)
</details>

---

**7.3*** Score and Langevin**
- (a) What's the score function for p(x) ∝ exp(-E(x))?
- (b) Rewrite Langevin using score instead of energy
- (c) Why is this connection important for diffusion?

<details>
<summary>Solution</summary>

(a) Score = ∇log p(x) = ∇(-E(x) - log Z) = -∇E(x)

(b) dx = score(x)dt + √(2/β)dW

(c) Diffusion models learn the score directly! Then use Langevin to sample.
</details>

---

### Coding Exercises

**7.4** Langevin Sampler**
```python
def langevin_sample(grad_log_p, x0, step_size, n_steps, temperature=1.0):
    """
    Sample from p(x) ∝ exp(log_p(x)) using Langevin dynamics.

    Args:
        grad_log_p: function computing ∇log p(x)
        x0: initial point
        step_size: dt
        n_steps: number of steps
        temperature: inverse β

    Returns:
        trajectory of samples
    """
    # Your code here
    pass

# Test: Sample from N(0, 1)
# grad_log_p(x) = -x for standard normal
samples = langevin_sample(lambda x: -x, 5.0, 0.1, 10000)
assert np.abs(samples[-1000:].mean()) < 0.1
assert np.abs(samples[-1000:].std() - 1.0) < 0.1
```

---

## Chapter 8: Energy-Based Models

### Conceptual Problems

**8.1* VAE Components**
- (a) What does the encoder output?
- (b) What's the reparameterization trick?
- (c) Why not just sample z ~ q(z|x) directly?

<details>
<summary>Solution</summary>

(a) Parameters of q(z|x): mean μ(x) and variance σ²(x)

(b) z = μ + σ ⊙ ε where ε ~ N(0,I). Moves randomness outside the network.

(c) Sampling is not differentiable! Can't backprop through random sampling. Reparameterization makes gradient flow through μ and σ.
</details>

---

**8.2** ELBO Terms**
For ELBO = E[log p(x|z)] - KL(q||p):
- (a) What happens if we maximize only reconstruction?
- (b) What happens if we only minimize KL?
- (c) How does β-VAE balance these?

<details>
<summary>Solution</summary>

(a) q(z|x) can be anything - posterior collapse to deterministic encoder, poor generation

(b) q(z|x) = p(z) = N(0,I) for all x - ignores input, poor reconstruction

(c) ELBO = reconstruction - β × KL. β > 1: more regularized, disentangled latents. β < 1: better reconstruction.
</details>

---

### Coding Exercises

**8.3** VAE Loss**
```python
def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    """
    Compute VAE ELBO loss.

    Args:
        x: original input
        x_recon: reconstructed output
        mu: encoder mean
        log_var: encoder log variance
        beta: KL weight

    Returns:
        total loss, reconstruction loss, KL loss
    """
    # Reconstruction: -E[log p(x|z)]
    # KL: KL(N(mu, var) || N(0, 1))
    # Your code here
    pass
```

---

## Chapter 9: Diffusion Models

### Conceptual Problems

**9.1* Forward Process**
For forward SDE: dx = -0.5β(t)x dt + √β(t) dW
- (a) What happens to x as t → ∞?
- (b) Why is this called "diffusion"?
- (c) What's the marginal distribution q(x_t|x_0)?

<details>
<summary>Solution</summary>

(a) x converges to N(0, I) - all structure destroyed

(b) Information "diffuses" away like heat spreading, data becomes noise

(c) q(x_t|x_0) = N(√α̅_t x_0, (1-α̅_t)I) where α̅_t = exp(-∫₀ᵗ β(s)ds)
</details>

---

**9.2** Score Matching**
- (a) What is the score function?
- (b) Write the denoising score matching objective
- (c) Why "denoising"?

<details>
<summary>Solution</summary>

(a) Score s(x,t) = ∇_x log p(x,t)

(b) L = E_{x₀,t,ε}[||s_θ(x_t, t) - ∇log p(x_t|x_0)||²]
    = E[||s_θ(x_t, t) + (x_t - x_0)/σ_t²||²]
    = E[||s_θ(x_t, t) + ε/σ_t||²] (since x_t = x_0 + σ_t ε)

(c) The score points toward the clean data x_0 - predicting how to denoise!
</details>

---

**9.3*** Unified View**
Explain how each model fits the diffusion framework:
- (a) VAE as diffusion
- (b) GAN as diffusion
- (c) Flow as diffusion

<details>
<summary>Solution</summary>

(a) **VAE**: One-step forward (encoder adds noise to get z), one-step reverse (decoder). Forward noise is learned, not fixed schedule.

(b) **GAN**: T=1, zero noise (g=0). Generator is a deterministic map from z to x. Equivalent to optimal transport / Schrödinger bridge with ε→0.

(c) **Flow**: Deterministic (no noise), ODE instead of SDE. dx/dt = f(x,t), no dW term. Probability flow ODE of diffusion.
</details>

---

### Coding Exercises

**9.4** Simple Diffusion**
```python
def forward_diffusion(x0, t, beta_schedule):
    """
    Sample x_t from q(x_t|x_0).

    Args:
        x0: clean data
        t: time step (0 to T)
        beta_schedule: array of beta values

    Returns:
        x_t, noise (epsilon)
    """
    # Compute alpha_bar_t
    # Sample noise
    # Return x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
    pass

def reverse_step(xt, t, score_model, beta_schedule):
    """
    One step of reverse diffusion.

    Args:
        xt: noisy sample at time t
        t: current time
        score_model: trained score network
        beta_schedule: noise schedule

    Returns:
        x_{t-1}
    """
    # Predict noise using score model
    # Apply reverse formula
    pass
```

---

**9.5*** Full Diffusion Training Loop**
```python
def train_diffusion(dataloader, score_model, optimizer, n_epochs, beta_schedule):
    """
    Train a diffusion model.

    For each batch:
    1. Sample random t
    2. Add noise to get x_t
    3. Predict noise with score_model
    4. Compute MSE loss
    """
    for epoch in range(n_epochs):
        for x0 in dataloader:
            # Your code here
            pass
```

---

## Challenge Problems (Integration)

### C1: From SVD to VAE
Design an experiment that:
1. Applies PCA (SVD) to MNIST
2. Trains an autoencoder with same latent dimension
3. Trains a VAE with same latent dimension
4. Compares reconstruction quality and latent space properties

---

### C2: Langevin to Diffusion
1. Implement Langevin sampling for a 2D Gaussian mixture
2. Implement a simple diffusion model for the same distribution
3. Compare sample quality and efficiency
4. Show how diffusion = annealed Langevin

---

### C3: Build a Tiny Diffusion Model
From scratch, build a diffusion model for 2D data:
1. Implement forward noising
2. Train a small MLP to predict score
3. Implement reverse sampling
4. Generate and visualize samples

---

*These problems accompany STUDY_GUIDE.md, MINDMAP.md, and NOTEBOOK_SEQUENCE.md*
