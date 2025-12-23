# Notebook Learning Sequence

> **Goal**: A structured path through all 64 notebooks with clear learning objectives
> **Estimated Total Time**: 40-50 hours of hands-on work

---

## How to Use This Guide

Each notebook entry includes:
- **Time**: Estimated completion time
- **Prerequisites**: What you should know first
- **Learning Objectives**: What you'll understand after
- **Key Experiments**: Modifications to try
- **Checkpoint Questions**: Verify understanding

---

## Phase 1: Foundations (Weeks 1-3)

### Week 1: Linear Algebra in Action

#### Lab 1.1: Image as Matrix
**Notebook**: `notebook/chapter1/LinearAlgebra/Ex1-1-1image.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Understand images as 3D tensors (H × W × 3)
- [ ] Extract and manipulate RGB channels
- [ ] Compute basic statistics (mean intensity)

**Key Experiments**:
```python
# Try these modifications:
1. Load your own image
2. Swap R and B channels - what happens?
3. Convert to grayscale using weighted average
```

**Checkpoint**: Can you explain why an RGB image is a rank-3 tensor?

---

#### Lab 1.2: SVD for Image Compression
**Notebook**: `notebook/chapter1/LinearAlgebra/MNIST-SVD.ipynb`
**Time**: 1 hour

**Prerequisites**: Lab 1.1

**Learning Objectives**:
- [ ] Perform SVD on image batches
- [ ] Visualize singular values and their decay
- [ ] Reconstruct images with varying ranks
- [ ] Understand the compression/quality tradeoff

**Key Experiments**:
```python
# Modify these parameters:
1. Change k from 5 to 200 - plot reconstruction error
2. Compare 90% vs 99% energy retention
3. Visualize top singular vectors ("eigendigits")
```

**Checkpoint**:
- Why do singular values decay?
- How many components capture 90% of variance?

---

### Week 2: Calculus and Dynamics

#### Lab 2.1: Automatic Differentiation
**Notebook**: `notebook/chapter2/AD/AD.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Understand forward vs reverse mode AD
- [ ] Trace computation graphs
- [ ] See how PyTorch computes gradients

**Key Experiments**:
```python
# Try:
1. Build a simple neural network, print gradients
2. Compare manual gradient vs autograd
3. Time forward vs reverse mode for different shapes
```

**Checkpoint**: Why is reverse mode better for neural networks?

---

#### Lab 2.2: Linear Systems and Heat Equation
**Notebook**: `notebook/chapter2/ODEs/linear_systems.ipynb`
**Time**: 45 min

**Prerequisites**: Lab 2.1

**Learning Objectives**:
- [ ] Solve linear ODEs numerically
- [ ] Visualize heat diffusion dynamics
- [ ] Connect to diffusion models conceptually

**Key Experiments**:
```python
# Modify:
1. Change diffusion coefficient - observe speed
2. Different initial conditions (spike, step, sine)
3. Compare explicit vs implicit solvers
```

**Checkpoint**: How does heat diffusion relate to adding noise in diffusion models?

---

#### Lab 2.3: Graph Dynamics
**Notebook**: `notebook/chapter2/ODEs/graph_dynamics.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Understand graph Laplacian
- [ ] See consensus/averaging dynamics
- [ ] Connect to GNNs and message passing

**Key Experiments**:
```python
# Try:
1. Different graph topologies (ring, star, complete)
2. Varying edge weights
3. Observe convergence speed vs connectivity
```

---

#### Lab 2.4: Double-Well Potential
**Notebook**: `notebook/chapter2/regression/double-well.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Visualize energy landscapes
- [ ] Understand metastability
- [ ] Preview Langevin dynamics

**Key Experiments**:
```python
# Explore:
1. Vary barrier height
2. Add small noise - observe transitions
3. Connect to sampling problems
```

**Checkpoint**: Why is it hard to sample from multimodal distributions?

---

### Week 3: Optimization

#### Lab 3.1: Logistic Regression Landscape
**Notebook**: `notebook/chapter3/Optimization/LogReg2D.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Visualize decision boundaries
- [ ] Understand convex loss landscape
- [ ] See gradient descent convergence

---

#### Lab 3.2: Gradient Field Visualization
**Notebook**: `notebook/chapter3/Optimization/LR-gradient-field.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] See gradients as vector fields
- [ ] Understand why GD follows contours
- [ ] Visualize learning rate effects

---

#### Lab 3.3: Convex vs Non-Convex
**Notebook**: `notebook/chapter3/Optimization/Convex_vs_NonConvex_Landscapes.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Distinguish convex from non-convex problems
- [ ] Understand local minima, saddle points
- [ ] See why neural nets are hard to optimize

---

#### Lab 3.4: Optimizer Comparison
**Notebook**: `notebook/chapter3/Optimization/adaptive.ipynb`
**Time**: 1 hour

**Prerequisites**: Labs 3.1-3.3

**Learning Objectives**:
- [ ] Compare SGD, Momentum, Adam, RMSProp
- [ ] Understand adaptive learning rates
- [ ] See bias correction in action

**Key Experiments**:
```python
# Compare:
1. Same LR for all optimizers - who wins?
2. Tune LR for each - how much work?
3. Add noise to gradients - robustness?
```

**Checkpoint**: Why is Adam the default choice for most problems?

---

#### Lab 3.5: Tiny Transformer
**Notebook**: `notebook/chapter3/Optimization/tiny-transformer.ipynb`
**Time**: 1.5 hours

**Learning Objectives**:
- [ ] See attention mechanism in code
- [ ] Understand positional encoding
- [ ] Train a small language model

**Key Experiments**:
```python
# Explore:
1. Visualize attention patterns
2. Remove positional encoding - what breaks?
3. Try different sequence lengths
```

---

## Phase 2: Neural Architectures (Week 4)

#### Lab 4.1: Simple CNN
**Notebook**: `notebook/chapter4/NN/cnn-simple-MNIST.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Build a basic CNN
- [ ] Understand convolution, pooling, stride
- [ ] Visualize learned filters

---

#### Lab 4.2: CNN with PCA Analysis
**Notebook**: `notebook/chapter4/NN/CNN-MNIST-PCA.ipynb`
**Time**: 1 hour

**Prerequisites**: Lab 1.2, Lab 4.1

**Learning Objectives**:
- [ ] Extract intermediate features
- [ ] Apply PCA to feature maps
- [ ] Visualize learned representations

**Checkpoint**: How do CNN features differ from raw pixels?

---

#### Lab 4.3: ResNet on Spiral
**Notebook**: `notebook/chapter4/NN/ResNet9-Spiral.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Implement skip connections
- [ ] See why depth helps
- [ ] Visualize decision boundaries evolving

---

#### Lab 4.4: Neural ODE on Spiral
**Notebook**: `notebook/chapter4/NN/NeuralODE-Spiral.ipynb`
**Time**: 1.5 hours

**Prerequisites**: Lab 4.3

**Learning Objectives**:
- [ ] Understand continuous-depth networks
- [ ] See ODE solver in action
- [ ] Compare to discrete ResNet

**Key Experiments**:
```python
# Compare:
1. Neural ODE vs ResNet-9 vs ResNet-18
2. Visualization of continuous trajectories
3. Memory usage during training
```

**Checkpoint**: What's the memory advantage of Neural ODE?

---

#### Lab 4.5: Adjoint Method
**Notebook**: `notebook/chapter4/NN/NeuralODE-Adjoint-Spiral.ipynb`
**Time**: 1 hour

**Prerequisites**: Lab 4.4

**Learning Objectives**:
- [ ] Understand adjoint sensitivity method
- [ ] See O(1) memory backprop
- [ ] Trade compute for memory

---

## Phase 3: Probability and Statistics (Week 5)

#### Lab 5.1: Inverse CDF Sampling
**Notebook**: `notebook/chapter5/Statistics/Inverse-CDF.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Implement inverse transform sampling
- [ ] Sample from custom distributions
- [ ] Understand when this works/doesn't

---

#### Lab 5.2: 1D Transformations
**Notebook**: `notebook/chapter5/Statistics/Transformations-1D.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Change of variables formula
- [ ] Jacobian for 1D case
- [ ] Connect to normalizing flows

---

#### Lab 5.3: Normalizing Flow 1D
**Notebook**: `notebook/chapter5/Statistics/Normalizing-Flow-1D.ipynb`
**Time**: 1.5 hours

**Prerequisites**: Labs 5.1-5.2

**Learning Objectives**:
- [ ] Build a simple normalizing flow
- [ ] Train via maximum likelihood
- [ ] Visualize density transformation

**Key Experiments**:
```python
# Try:
1. Different target distributions (bimodal, skewed)
2. More flow layers
3. Compute exact likelihoods
```

**Checkpoint**: How does flow compute exact likelihood unlike VAE?

---

#### Lab 5.4: Empirical Distributions
**Notebook**: `notebook/chapter5/Statistics/Empirical-Distributions-1D.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Build empirical CDFs
- [ ] Kernel density estimation
- [ ] Compare to true distribution

---

#### Lab 5.5: Multivariate Analysis
**Notebook**: `notebook/chapter5/Statistics/Empirical-Multivariate.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Covariance matrices
- [ ] Correlation structure
- [ ] Mahalanobis distance

---

## Phase 4: Information Theory (Week 6)

#### Lab 6.1: Bayesian Toy Example
**Notebook**: `notebook/chapter6/Info/Bayes-Toy-Discrete.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Compute posterior manually
- [ ] Understand prior × likelihood ∝ posterior
- [ ] Medical diagnosis example

---

#### Lab 6.2: Naive Bayes on MNIST
**Notebook**: `notebook/chapter6/Info/Bayes-MNIST-NaiveBayes.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Implement Naive Bayes classifier
- [ ] Compare to neural network
- [ ] Understand independence assumption

**Checkpoint**: Why does Naive Bayes work despite wrong assumptions?

---

#### Lab 6.3: Entropy in Classification
**Notebook**: `notebook/chapter6/Info/Entropy-Classification-Experiment.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Compute predictive entropy
- [ ] Identify uncertain samples
- [ ] Understand model confidence

**Key Experiments**:
```python
# Analyze:
1. High vs low entropy predictions
2. Correlation with correctness
3. Use for active learning
```

---

#### Lab 6.4: Autoencoder with Entropy
**Notebook**: `notebook/chapter6/Info/autoencoder-entropy.ipynb`
**Time**: 1.5 hours

**Prerequisites**: Phase 2

**Learning Objectives**:
- [ ] Build autoencoder
- [ ] Information bottleneck intuition
- [ ] Reconstruction vs compression tradeoff

---

#### Lab 6.5: Wasserstein Distance
**Notebook**: `notebook/chapter6/Info/Wasserstein-1D-Gaussians.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Compute Wasserstein distance
- [ ] Compare to KL divergence
- [ ] Understand optimal transport basics

---

## Phase 5: Stochastic Processes (Weeks 7-8)

#### Lab 7.1: Inverse Transform Sampling
**Notebook**: `notebook/chapter7/Stochastic/ITS-1D.ipynb`
**Time**: 30 min

**Learning Objectives**:
- [ ] Review sampling fundamentals
- [ ] Prepare for MCMC

---

#### Lab 7.2: Chain Rule Sampling
**Notebook**: `notebook/chapter7/Stochastic/ChainRuleSampling-2D.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Sequential sampling p(x,y) = p(x)p(y|x)
- [ ] Autoregressive modeling intuition

---

#### Lab 7.3: Importance Sampling Basics
**Notebook**: `notebook/chapter7/Stochastic/ImportanceSampling-RareEvent.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Reweight samples from proposal
- [ ] Handle rare events
- [ ] Variance reduction

---

#### Lab 7.4: Importance Sampling for Inference
**Notebook**: `notebook/chapter7/Stochastic/ImportanceSampling-GaussianPosterior.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Approximate intractable posteriors
- [ ] Effective sample size
- [ ] Connection to VAE importance weighting

---

#### Lab 7.5: Brownian Motion & Heat Equation
**Notebook**: `notebook/chapter7/Stochastic/BrownianMotion-and-HeatEquation.ipynb`
**Time**: 1.5 hours

**Learning Objectives**:
- [ ] Simulate Brownian paths
- [ ] Connect to diffusion PDE
- [ ] Foundational for diffusion models

**Key Experiments**:
```python
# Visualize:
1. Many Brownian paths
2. Distribution evolution over time
3. Connection to Gaussian blur
```

**Checkpoint**: Why does the distribution become Gaussian?

---

#### Lab 7.6: Langevin Dynamics (CRITICAL)
**Notebook**: `notebook/chapter7/Stochastic/Langevin-DoubleWell.ipynb`
**Time**: 2 hours

**Prerequisites**: Labs 7.1-7.5

**Learning Objectives**:
- [ ] Implement Langevin sampler
- [ ] See particles settling into energy minima
- [ ] Understand temperature/noise tradeoff
- [ ] **Direct connection to diffusion reverse process**

**Key Experiments**:
```python
# Critical experiments:
1. Vary temperature - see mode hopping
2. Remove noise - particles get stuck
3. Visualize trajectory in energy landscape
4. Compute empirical distribution vs theory
```

**Checkpoint**:
- Why do we need noise for sampling?
- How does Langevin connect to score-based diffusion?

---

#### Lab 7.7: RBM with MCMC
**Notebook**: `notebook/chapter7/Stochastic/RBM-MCMC.ipynb`
**Time**: 1.5 hours

**Learning Objectives**:
- [ ] Implement Gibbs sampling for RBM
- [ ] Energy-based model sampling
- [ ] Contrastive divergence training

---

#### Lab 7.8: Adaptive Importance Sampling
**Notebook**: `notebook/chapter7/Stochastic/AdaptiveIS-CE-fitGaussian.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Cross-entropy method
- [ ] Iteratively improve proposal
- [ ] Connection to variational inference

---

## Phase 6: Energy-Based Models (Week 9)

#### Lab 8.1: Variational Inference Toy
**Notebook**: `notebook/chapter8/VI_ELBO_Toy.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Implement ELBO optimization
- [ ] See q(z) approach true posterior
- [ ] Understand approximation gap

---

#### Lab 8.2: Mean Field Ising
**Notebook**: `notebook/chapter8/MeanField_Ising_3x3.ipynb`
**Time**: 45 min

**Learning Objectives**:
- [ ] Mean field approximation
- [ ] Factorized distributions
- [ ] Connection to variational methods

---

#### Lab 8.3: Belief Propagation
**Notebook**: `notebook/chapter8/BP_Tree_vs_Loopy_Ising.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Message passing algorithms
- [ ] Tree vs loopy graphs
- [ ] Connection to GNN

---

#### Lab 8.4: Score Matching for Energy
**Notebook**: `notebook/chapter8/ScoreMatchingEnergy.ipynb`
**Time**: 1.5 hours

**Prerequisites**: Lab 7.6

**Learning Objectives**:
- [ ] Learn energy function via score matching
- [ ] Avoid partition function
- [ ] **Direct setup for diffusion**

---

#### Lab 8.5: Ring VAE
**Notebook**: `notebook/chapter8/Ring_VAE_1D_Latent.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] VAE on simple distribution
- [ ] Visualize latent space
- [ ] Reconstruction vs KL tradeoff

---

#### Lab 8.6: GNN vs Graphical Models
**Notebook**: `notebook/chapter8/gnn_vs_gm_grid.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] Compare inference methods
- [ ] GNN as learned message passing
- [ ] When to use which

---

## Phase 7: Diffusion Models (Weeks 10-12) - CAPSTONE

#### Lab 9.1: Score-Based Diffusion on Grid
**Notebook**: `notebook/chapter9/02-SGM-with-SDE-9grid.ipynb`
**Time**: 3 hours

**Prerequisites**: All previous phases, especially 7.6 and 8.4

**Learning Objectives**:
- [ ] Implement forward noising process
- [ ] Train score network
- [ ] Run reverse SDE for generation
- [ ] Visualize the full pipeline

**Key Experiments**:
```python
# Essential experiments:
1. Visualize forward process (data → noise)
2. Visualize learned score field
3. Generate samples with different step counts
4. Compare to ground truth distribution
```

**Checkpoint**:
- What does the score network learn?
- Why do we need many denoising steps?
- How does this connect to Langevin dynamics?

---

#### Lab 9.2: Schrödinger Bridge
**Notebook**: `notebook/chapter9/03-SchrBridge-1D.ipynb`
**Time**: 1.5 hours

**Learning Objectives**:
- [ ] Bridge diffusion concept
- [ ] Finite-time diffusion
- [ ] Connection to optimal transport

---

#### Lab 9.3: GAN as Transport
**Notebook**: `notebook/chapter9/gan_as_deterministic_transport.ipynb`
**Time**: 1 hour

**Learning Objectives**:
- [ ] GAN as deterministic map
- [ ] Comparison to stochastic diffusion
- [ ] Unified view of generative models

---

#### Lab 9.4: VAE vs Latent Diffusion
**Notebook**: `notebook/chapter9/ring_vae_latent_diffusion.ipynb`
**Time**: 1.5 hours

**Learning Objectives**:
- [ ] VAE = one-step diffusion
- [ ] Latent diffusion concept
- [ ] Quality comparison

**Checkpoint**:
- How are VAE, GAN, and Diffusion related?
- What's the tradeoff between them?

---

## Completion Checklist

### Phase Completion Markers

- [ ] **Phase 1 Complete**: Can explain SVD, whitening, and backprop
- [ ] **Phase 2 Complete**: Can implement CNN and understand ResNet
- [ ] **Phase 3 Complete**: Can explain normalizing flows
- [ ] **Phase 4 Complete**: Can derive ELBO and explain KL divergence
- [ ] **Phase 5 Complete**: Can implement Langevin sampler
- [ ] **Phase 6 Complete**: Can explain VAE and energy-based models
- [ ] **Phase 7 Complete**: Can explain and implement basic diffusion model

### Final Integration Questions

After completing all phases, you should be able to answer:

1. **Trace the math**: How does data flow from input to generated output in a diffusion model?

2. **Unified view**: How are VAE, GAN, Flow, and Diffusion related mathematically?

3. **Design decisions**: Why do diffusion models use many steps? Why U-Net?

4. **Practical tradeoffs**: When would you choose each generative model type?

---

## Quick Reference: Notebook Dependencies

```
Lab 1.1 ─────────────────────────────────────────┐
   │                                              │
Lab 1.2 ───────────────────┬──────────────────────┼──→ Lab 4.2
                           │                      │
Lab 2.1 → Lab 2.2 → Lab 2.4                       │
   │                 │                            │
Lab 3.1-3.4 ─────────┴──→ Lab 3.5                 │
   │                                              │
Lab 4.1 → Lab 4.3 → Lab 4.4 → Lab 4.5             │
                       │                          │
Lab 5.1-5.2 → Lab 5.3  │                          │
                 │     │                          │
Lab 6.1-6.3 ─────┴─────┴──→ Lab 6.4               │
                              │                   │
Lab 7.1-7.5 → Lab 7.6 ────────┴──→ Lab 8.4        │
                │                    │            │
Lab 8.1-8.3 ────┴──→ Lab 8.5        │            │
                       │             │            │
                       └──→ Lab 9.1 ←┴────────────┘
                              │
                    Lab 9.2, 9.3, 9.4
```

---

*This sequence guide accompanies STUDY_GUIDE.md and MINDMAP.md*
