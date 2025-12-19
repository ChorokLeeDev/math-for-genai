# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 📚 Repository Overview

**Mathematics of Generative AI Book** - A living textbook with computational Jupyter notebooks organized by mathematical topics for understanding generative AI from first principles.

### Key Characteristics
- **Notebook-heavy**: Jupyter notebooks (.ipynb) are the primary educational medium
- **Mathematics-first approach**: Each chapter builds mathematical foundations for generative AI
- **Live implementations**: PyTorch/NumPy code alongside theory
- **Large PDF reference**: MathGenAIBook12_14_25.pdf (~14.5 MB) - use chapter page ranges to navigate

---

## 📖 PDF Chapter Guide (CRITICAL for Context Management)

**When asked about a chapter, use ONLY the relevant page range. Do NOT read the entire PDF.**

| Chapter | Topic | Pages | Key Concepts |
|---------|-------|-------|--------------|
| **1** | Linear Algebra (of AI) | 10-36 | SVD, Eigendecomposition, Convolution, Tensors, Transformers |
| **2** | Calculus & Differential Equations | 38-63 | AD (Forward/Reverse), ODEs, System of Linear ODEs, Graph Dynamics |
| **3** | Optimization (in AI) | 65-97 | Gradient Descent, Convex Optimization, Regularization, SGD variants |
| **4** | Neural Networks & Deep Learning | 99-132 | NN Mechanics, CNN, ResNet, Neural ODEs, Manifold Selection |
| **5** | Probability and Statistics | 134-162 | Probability Spaces, Normalizing Flows, Multivariate Gaussian, CLT |
| **6** | Entropy and Information Theory | 163-196 | Bayes' Rule, Entropy, KL Divergence, Autoencoders, Information Bottleneck |
| **7** | Stochastic Processes | 198-233 | Importance Sampling, Diffusion, Brownian Motion, MCMC, Auto-Regressive |
| **8** | Energy Based (Graphical) Models | 235-264 | Graphical Models, VAE, RBM, Graph Neural Networks |
| **9** | Synthesis | 266+ | Score-Based Diffusion, GANs, Phase Transitions, RL, Generative Flow Networks |

---

## 🗂️ Repository Structure

```
notebook/
├── chapter1/LinearAlgebra/
│   ├── MNIST-SVD.ipynb           (SVD visualization with handwritten digits)
│   └── Ex1-1-1image.ipynb        (Simple image matrix decomposition)
├── chapter2/                       (Calculus notebooks - ODE solvers, AD examples)
├── chapter3/                       (Optimization - gradient descent, regularization)
├── chapter4/                       (Neural networks - CNN, ResNet, Neural ODE)
├── chapter5/Statistics/            (Distributions, flows, empirical statistics)
├── chapter6/Info/                  (Entropy, autoencoders, information theory)
├── chapter7/Stochastic/            (Diffusion, Brownian motion, MCMC, Langevin)
└── chapter8+/                      (Energy models, synthesis)

MathGenAIBook12_14_25.pdf          (Main reference textbook)
```

---

## 🎯 When Asked About a Chapter

### **Question: "Explain SVD" or "Help with Chapter 1"**

**DO THIS:**
1. Read `notebook/chapter1/README.md` (if exists)
2. Check relevant notebooks (MNIST-SVD.ipynb, Ex1-1-1image.ipynb)
3. For detailed math: "Which section of Chapter 1 (pages 10-36)?"
4. If needed: Read ONLY pages 10-36 of PDF

**DON'T:**
- Read entire PDF at once
- Extract large PDF sections without asking which subsection
- Assume you know the structure without checking README.md first

### **Question: "What's in the PDF?"**

**RESPONSE PATTERN:**
```
"This repository covers 9 chapters (pages 10-end).
Which chapter interests you?
- Ch1: SVD, Eigendecomposition (pages 10-36)
- Ch2: Automatic Differentiation, ODEs (pages 38-63)
- Ch6: KL Divergence, VAE (pages 163-196)
..."
```

Then read ONLY that chapter's pages or notebook files.

---

## 📝 Content Structure: Definition → Intuition → Application

Each chapter should follow:

1. **Definition (수학적 정의)**
   - Formal mathematical statement
   - Notation and symbols clearly explained

2. **Intuition (직관적 이해)**
   - Geometric interpretation
   - Physical analogy or visualization
   - Why it matters conceptually

3. **AI Application (AI에서의 응용)**
   - How this is used in generative AI
   - Connection to VAE, Diffusion, Transformers, etc.

**Example:** SVD should explain:
- Definition: $X = U\Sigma V^T$
- Intuition: "Rotation → Scaling → Rotation" or "Finding important directions"
- Application: Data compression, PCA, denoising

---

## 💻 Working with Notebooks

### Reading Notebooks
```bash
# See chapter structure:
ls notebook/chapterN/*/

# Read full notebook (includes metadata):
# Use Read tool on .ipynb files
```

### Editing Notebooks
```bash
# DO: Use NotebookEdit tool
# DON'T: Use Edit tool on .ipynb (JSON format, breaks cells)

# Example (adding explanation cell):
NotebookEdit(
  notebook_path="/path/to/notebook.ipynb",
  cell_number=5,
  new_source="# SVD decomposes the matrix as X = U Σ V^T",
  edit_mode="insert",
  cell_type="markdown"
)
```

### Notebook Execution
- Notebooks are self-contained educational units
- They should run end-to-end without errors
- Include both mathematical notation AND code visualization

---

## 🎓 Improving Chapter Documentation

### README.md Pattern (Each chapter should have)

```markdown
# Chapter N: [Topic]

## Core Concepts
- Define each concept (not just list)
- Explain notation

## Intuition Section
- Geometric/physical interpretation
- Visualizations or analogies
- Why this matters

## AI Applications
- 2-3 specific use cases
- Connection to generative AI models (VAE, Diffusion, Transformer, GAN)

## Key Notebook Guide
- What each notebook teaches
- How they connect to the chapter content
```

### When Enhancing Content

1. **Check notebook code first** - don't just cite PDF
2. **Add visual interpretations** - describe what the math looks like
3. **Connect to next chapter** - "This is used in Chapter X for..."
4. **Emphasize AI relevance** - "This is why we need this in generative models"

---

## 🔄 Development Workflow

### Working on Chapter Improvements

```
1. Ask which chapter (e.g., "Chapter 1: Linear Algebra, pages 10-36")
2. Read: notebook/chapter1/README.md
3. Read: relevant notebooks (MNIST-SVD.ipynb, etc.)
4. Check PDF pages 10-36 if needed
5. Improve README or notebooks
6. Commit with message: "Improve Chapter N: [specific improvement]"
```

### Git Commit Messages

**IMPORTANT:** Do NOT include the Claude Code footer in commit messages. No "🤖 Generated with Claude Code" or "Co-Authored-By: Claude" lines.

```bash
# When updating chapter docs:
git commit -m "Improve Chapter 1 README: Add SVD-Gaussian-Whitening connection"

# When modifying notebooks:
git commit -m "Update Chapter 6 notebook: Clarify KL Divergence penalty terms"

# When adding new content:
git commit -m "Add Chapter 2 intuition: Forward vs Reverse AD complexity comparison"
```

### Git Remotes

**IMPORTANT:** Do NOT push to `origin` remote. Only push to `chorok` remote.

**IMPORTANT:** Always pull and merge before pushing. Never force push.

```bash
# Before pushing, always pull first:
git pull chorok main
git push chorok main

# NEVER do this:
git push origin main      # ❌ Do not push to origin
git push --force          # ❌ Never force push
```

---

## 🛠️ Technical Notes

### No Build/Test Framework
- Educational repository (no CI/CD, no unit tests, no linting)
- Notebooks are the "tests" - they should run without errors
- Git is the only infrastructure

### Dependencies (Typical)
- Jupyter/IPython
- NumPy
- PyTorch
- Matplotlib
- (Check individual notebooks for specifics)

### No Local Setup Required
- Notebooks are self-contained
- Can be viewed in GitHub or Jupyter locally
- No build step needed

---

## 📋 Quick Reference: What's in Each Notebook Folder?

| Folder | Purpose | Example Notebooks |
|--------|---------|-------------------|
| `chapter1/LinearAlgebra/` | SVD, matrix decomposition | MNIST-SVD, Ex1-1-1image |
| `chapter5/Statistics/` | Distributions, normalizing flows | Normalizing-Flow-1D, Empirical-Multivariate |
| `chapter6/Info/` | Entropy, VAE, CNN | autoencoder-entropy, CNN-MNIST-channel |
| `chapter7/Stochastic/` | Diffusion, MCMC, sampling | Langevin-DoubleWell, BrownianMotion-HeatEquation |

---

## 💡 Context Management Strategy

### Handling Large Files

**PDF (14.5 MB):**
- Use chapter page ranges from table above
- Read ONLY the relevant pages
- Always ask "which section?" if unclear

#### PDF 읽기 방법 (PyMuPDF 사용)

Claude Code의 Read 도구는 대용량 PDF를 직접 읽을 수 없습니다. 대신 PyMuPDF를 사용하세요:

```python
# 1. PyMuPDF 설치 (최초 1회)
pip3 install pymupdf

# 2. 특정 챕터 페이지만 추출
python3 << 'EOF'
import fitz  # PyMuPDF

doc = fitz.open("MathGenAIBook12_14_25.pdf")

# Chapter 1: pages 10-36 (0-indexed: 9-35)
for page_num in range(9, 36):
    page = doc[page_num]
    text = page.get_text()
    print(f"\n--- Page {page_num + 1} ---\n")
    print(text)

doc.close()
EOF
```

**페이지 범위 참조 (0-indexed):**
| Chapter | Pages (1-indexed) | range() (0-indexed) |
|---------|-------------------|---------------------|
| 1 | 10-36 | `range(9, 36)` |
| 2 | 38-63 | `range(37, 63)` |
| 3 | 65-97 | `range(64, 97)` |
| 4 | 99-132 | `range(98, 132)` |
| 5 | 134-162 | `range(133, 162)` |
| 6 | 163-196 | `range(162, 196)` |
| 7 | 198-233 | `range(197, 233)` |
| 8 | 235-264 | `range(234, 264)` |
| 9 | 266+ | `range(265, len(doc))` |

**Notebooks (1-3 MB each):**
- Self-contained, can read fully
- Use NotebookEdit for precise cell modifications

### When Working Across Multiple Chapters

1. **Focus on one chapter at a time** - complete improvements before moving to next
2. **Use TodoWrite** - track which chapters have been updated
3. **Cross-reference carefully** - "Chapter 2 uses concept from Chapter 1"

---

## 🎯 Example: Handling a Chapter 1 Question

**User asks:** "Why is SVD important for generative AI?"

**Claude should:**
1. ✅ Read notebook/chapter1/README.md
2. ✅ Check MNIST-SVD.ipynb to see code examples
3. ✅ Reference pages 27-36 (SVD section in PDF) if needed
4. ❌ NOT read entire PDF
5. ❌ NOT try to understand all 9 chapters at once

**Response includes:**
- Definition: "X = U Σ V^T decomposes into rotation-scale-rotation"
- Intuition: "Finds the k most important directions (low-rank approximation)"
- Application: "VAE uses this for dimensionality reduction; Diffusion models use it for whitening"
