#!/usr/bin/env python3
"""
Generate visual mind maps for Mathematics of Generative AI Book.
Creates PNG images that can be viewed directly.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up for better rendering - LARGER FONTS
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14  # Base font size increased
plt.rcParams['figure.dpi'] = 200  # Higher DPI for clarity
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 16

def create_master_mindmap():
    """Create the master mind map showing all chapters and their relationships."""
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    ax.set_title('Mathematics of Generative AI: Master Mind Map', fontsize=28, fontweight='bold', pad=30)

    # Color scheme
    colors = {
        'foundation': '#a8d5ff',  # Light blue
        'neural': '#a8ffa8',      # Light green
        'probability': '#ffa8ff', # Light magenta
        'generative': '#ffd700',  # Gold
        'diffusion': '#ff6b6b',   # Coral red (highlight)
    }

    # Define chapter boxes with positions - SCALED UP
    chapters = {
        # Foundations (left column)
        'Ch.1\nLinear Algebra': (3, 13, colors['foundation']),
        'Ch.2\nCalculus & ODE': (3, 9.5, colors['foundation']),
        'Ch.3\nOptimization': (3, 6, colors['foundation']),

        # Neural Networks (center-left)
        'Ch.4\nNeural Networks': (9, 9.5, colors['neural']),

        # Probability (center-right)
        'Ch.5\nProbability': (15, 13, colors['probability']),
        'Ch.6\nInformation Theory': (15, 9.5, colors['probability']),
        'Ch.7\nStochastic Processes': (15, 6, colors['probability']),

        # Generative Models (right)
        'Ch.8\nEnergy-Based Models': (21, 9.5, colors['generative']),

        # Diffusion (bottom center, highlighted)
        'Ch.9\nScore-Based\nDiffusion': (12, 2, colors['diffusion']),
    }

    # Draw boxes - LARGER
    boxes = {}
    for name, (x, y, color) in chapters.items():
        width, height = 4, 2.2
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                              boxstyle="round,pad=0.05,rounding_size=0.3",
                              facecolor=color, edgecolor='black', linewidth=3)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=15, fontweight='bold')
        boxes[name] = (x, y)

    # Draw arrows (connections)
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=6"
    kw = dict(arrowstyle=arrow_style, color="gray", lw=1.5,
              connectionstyle="arc3,rad=0.1")

    def add_arrow(start_name, end_name, rad=0.1):
        x1, y1 = boxes[start_name]
        x2, y2 = boxes[end_name]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle=arrow_style, color="gray", lw=1.5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 mutation_scale=15)
        ax.add_patch(arrow)

    # Foundations → Neural Networks
    add_arrow('Ch.1\nLinear Algebra', 'Ch.4\nNeural Networks', 0.15)
    add_arrow('Ch.2\nCalculus & ODE', 'Ch.4\nNeural Networks', 0)
    add_arrow('Ch.3\nOptimization', 'Ch.4\nNeural Networks', -0.15)

    # Probability chain
    add_arrow('Ch.5\nProbability', 'Ch.6\nInformation Theory', 0)
    add_arrow('Ch.6\nInformation Theory', 'Ch.7\nStochastic Processes', 0)

    # Cross connections
    add_arrow('Ch.1\nLinear Algebra', 'Ch.5\nProbability', 0.2)
    add_arrow('Ch.4\nNeural Networks', 'Ch.8\nEnergy-Based Models', 0.1)
    add_arrow('Ch.6\nInformation Theory', 'Ch.8\nEnergy-Based Models', -0.1)
    add_arrow('Ch.7\nStochastic Processes', 'Ch.8\nEnergy-Based Models', -0.2)

    # To Diffusion
    add_arrow('Ch.8\nEnergy-Based Models', 'Ch.9\nScore-Based\nDiffusion', -0.2)
    add_arrow('Ch.7\nStochastic Processes', 'Ch.9\nScore-Based\nDiffusion', 0.1)
    add_arrow('Ch.4\nNeural Networks', 'Ch.9\nScore-Based\nDiffusion', 0.2)

    # Add section labels - LARGER
    ax.text(3, 15.3, 'MATHEMATICAL\nFOUNDATIONS', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#0066cc',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
    ax.text(15, 15.3, 'PROBABILISTIC\nFOUNDATIONS', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#9900cc',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
    ax.text(21, 12.5, 'GENERATIVE\nMODELS', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#cc6600',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))

    # Add legend - LARGER
    legend_elements = [
        mpatches.Patch(facecolor=colors['foundation'], edgecolor='black', label='Math Foundations'),
        mpatches.Patch(facecolor=colors['neural'], edgecolor='black', label='Neural Networks'),
        mpatches.Patch(facecolor=colors['probability'], edgecolor='black', label='Probability'),
        mpatches.Patch(facecolor=colors['generative'], edgecolor='black', label='Energy Models'),
        mpatches.Patch(facecolor=colors['diffusion'], edgecolor='black', label='Diffusion (Goal)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=14)

    plt.tight_layout()
    plt.savefig('mindmaps/01_master_mindmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/01_master_mindmap.png")


def create_concept_flow():
    """Create detailed concept flow showing how specific topics connect."""
    fig, ax = plt.subplots(1, 1, figsize=(28, 18))
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 18)
    ax.axis('off')
    ax.set_title('Concept Flow: From Foundations to Diffusion', fontsize=28, fontweight='bold', pad=30)

    # Colors
    ch_colors = {
        1: '#cce5ff', 2: '#d4edda', 3: '#fff3cd',
        4: '#f8d7da', 5: '#e2d5f1', 6: '#d1ecf1',
        7: '#ffeeba', 8: '#c3e6cb', 9: '#f5c6cb'
    }

    # Chapter sections (x_start, x_end, y_start, y_end, chapter) - SCALED UP
    sections = [
        (0.5, 6.5, 13.5, 17.5, 1, 'Ch.1: Linear Algebra'),
        (0.5, 6.5, 9, 13, 2, 'Ch.2: Calculus'),
        (0.5, 6.5, 4.5, 8.5, 3, 'Ch.3: Optimization'),
        (7, 13, 9, 13, 4, 'Ch.4: Neural Nets'),
        (7, 13, 13.5, 17.5, 5, 'Ch.5: Probability'),
        (13.5, 19.5, 13.5, 17.5, 6, 'Ch.6: Information'),
        (13.5, 19.5, 9, 13, 7, 'Ch.7: Stochastic'),
        (13.5, 19.5, 4.5, 8.5, 8, 'Ch.8: Energy Models'),
        (20, 27, 9, 13, 9, 'Ch.9: DIFFUSION'),
    ]

    # Draw section backgrounds
    for x1, x2, y1, y2, ch, title in sections:
        rect = FancyBboxPatch((x1, y1), x2-x1, y2-y1,
                               boxstyle="round,pad=0.02,rounding_size=0.2",
                               facecolor=ch_colors[ch], edgecolor='gray',
                               linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text((x1+x2)/2, y2-0.4, title, ha='center', va='top',
                fontsize=16, fontweight='bold')

    # Concepts within each chapter - SCALED UP
    concepts = {
        # Chapter 1
        'SVD': (2, 16), 'Whitening': (4, 16), 'Attention\nQ,K,V': (5.5, 16),
        'Tensors': (2, 14.5), 'Einstein\nNotation': (4.5, 14.5),

        # Chapter 2
        'Auto-Diff': (2, 12), 'ODEs': (4, 12), 'Jacobian': (5.5, 12),
        'Chain\nRule': (2, 10), 'ResNet\n=Euler': (4.5, 10),

        # Chapter 3
        'Gradient\nDescent': (2, 7.5), 'Adam': (4, 7.5), 'L1/L2\nReg': (5.5, 7.5),
        'Loss\nLandscape': (3.5, 5.5),

        # Chapter 4
        'CNN': (8, 12), 'ResNet': (10, 12), 'Neural\nODE': (12, 12),
        'U-Net': (10, 10),

        # Chapter 5
        'Distributions': (8, 16.5), 'Normalizing\nFlow': (10, 16.5), 'Gaussian': (12, 16.5),
        'Sampling': (8, 14.5), 'Change of\nVariables': (11, 14.5),

        # Chapter 6
        'Entropy\nH(X)': (14.5, 16.5), 'KL\nDivergence': (16.5, 16.5), 'ELBO': (18.5, 16.5),
        'Cross\nEntropy': (14.5, 14.5), 'Mutual\nInfo': (17.5, 14.5),

        # Chapter 7
        'Brownian\nMotion': (14.5, 12), 'SDEs': (16.5, 12), 'Langevin': (18.5, 12),
        'MCMC': (14.5, 10), 'Fokker\nPlanck': (17.5, 10),

        # Chapter 8
        'Energy\nE(x)': (14.5, 7.5), 'VAE': (16.5, 7.5), 'RBM': (18.5, 7.5),
        'ELBO\nLoss': (14.5, 5.5), 'Variational\nInference': (17.5, 5.5),

        # Chapter 9 - Highlighted
        'Score\n∇log p': (21.5, 12), 'Forward\nSDE': (24, 12), 'Reverse\nSDE': (26, 12),
        'Denoising': (22.5, 10), 'Generation': (25, 10),
    }

    # Draw concept nodes - LARGER
    for name, (x, y) in concepts.items():
        color = 'white'
        if 'Score' in name or 'Forward' in name or 'Reverse' in name:
            color = '#ffcccc'
        box = FancyBboxPatch((x-1, y-0.6), 2, 1.2,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')

    # Key arrows showing flow
    arrows = [
        # SVD → Whitening → VAE
        ('SVD', 'Whitening', 0),
        ('Whitening', 'VAE', 0.3),

        # ODE → ResNet → Neural ODE → Forward SDE
        ('ODEs', 'ResNet\n=Euler', 0),
        ('ResNet', 'Neural\nODE', 0),
        ('Neural\nODE', 'Forward\nSDE', 0.2),

        # Auto-Diff → Gradient Descent → Adam
        ('Auto-Diff', 'Gradient\nDescent', 0.2),
        ('Gradient\nDescent', 'Adam', 0),

        # Gaussian → SDE
        ('Gaussian', 'SDEs', 0.2),

        # Entropy → KL → ELBO
        ('Entropy\nH(X)', 'KL\nDivergence', 0),
        ('KL\nDivergence', 'ELBO', 0),
        ('ELBO', 'VAE', 0.3),

        # Brownian → SDE → Langevin → Score
        ('Brownian\nMotion', 'SDEs', 0),
        ('SDEs', 'Langevin', 0),
        ('Langevin', 'Score\n∇log p', 0.2),

        # Score → Forward/Reverse
        ('Score\n∇log p', 'Reverse\nSDE', 0),
        ('Forward\nSDE', 'Reverse\nSDE', 0),

        # VAE → Score (one-step connection)
        ('VAE', 'Score\n∇log p', 0.3),
    ]

    for start, end, rad in arrows:
        if start in concepts and end in concepts:
            x1, y1 = concepts[start]
            x2, y2 = concepts[end]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#666666',
                                        lw=1.5, connectionstyle=f'arc3,rad={rad}'))

    plt.tight_layout()
    plt.savefig('mindmaps/02_concept_flow.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/02_concept_flow.png")


def create_unified_view():
    """Show how all generative models relate to diffusion."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Unified View: All Roads Lead to Diffusion', fontsize=28, fontweight='bold', pad=30)

    # Pre-diffusion models (left side) - LARGER
    models = {
        'VAE\n(2013)': (3.5, 11.5, '#a8d5ff'),
        'GAN\n(2014)': (3.5, 8.5, '#a8ffa8'),
        'Normalizing\nFlow (2015)': (3.5, 5.5, '#ffa8ff'),
        'Energy-Based\n(2019)': (3.5, 2.5, '#ffd700'),
    }

    # Interpretations (middle)
    interpretations = {
        'VAE = 1-step\ndiffusion': (10, 11.5, '#d4edff'),
        'GAN = deterministic\ntransport': (10, 8.5, '#d4ffd4'),
        'Flow = ODE\ndiffusion': (10, 5.5, '#ffd4ff'),
        'EBM = score\nfunction': (10, 2.5, '#ffeda0'),
    }

    # Final diffusion (right side)
    diffusion = ('Score-Based\nDIFFUSION\n(2020-now)', (16.5, 7, '#ff6b6b'))

    # Draw model boxes - LARGER
    for name, (x, y, color) in models.items():
        box = FancyBboxPatch((x-1.8, y-1), 3.6, 2,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=3)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=16, fontweight='bold')

    # Draw interpretation boxes - LARGER
    for name, (x, y, color) in interpretations.items():
        box = FancyBboxPatch((x-2.2, y-0.9), 4.4, 1.8,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='gray', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=14)

    # Draw diffusion box (highlighted) - LARGER
    name, (x, y, color) = diffusion
    box = FancyBboxPatch((x-2.5, y-2.2), 5, 4.4,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          facecolor=color, edgecolor='darkred', linewidth=5)
    ax.add_patch(box)
    ax.text(x, y, name, ha='center', va='center', fontsize=20, fontweight='bold', color='white')

    # Draw arrows
    # Models → Interpretations
    for (m_name, (mx, my, _)), (i_name, (ix, iy, _)) in zip(models.items(), interpretations.items()):
        ax.annotate('', xy=(ix-2.2, iy), xytext=(mx+1.8, my),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=3))

    # Interpretations → Diffusion
    for i_name, (ix, iy, _) in interpretations.items():
        ax.annotate('', xy=(16.5-2.5, 7), xytext=(ix+2.2, iy),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=3,
                                    connectionstyle='arc3,rad=0.1'))

    # Add labels - LARGER
    ax.text(3.5, 13.3, 'Pre-Diffusion Models', ha='center', fontsize=18, fontweight='bold')
    ax.text(10, 13.3, 'Diffusion Interpretation', ha='center', fontsize=18, fontweight='bold')
    ax.text(16.5, 13.3, 'Unified Framework', ha='center', fontsize=18, fontweight='bold', color='darkred')

    # Add key insight box - LARGER
    insight_text = """KEY INSIGHT:
All generative models can be viewed as
special cases of the diffusion framework!

• VAE: One encoding + decoding step
• GAN: Zero noise, deterministic map
• Flow: ODE instead of SDE
• EBM: Score is energy gradient"""

    ax.text(10, 0.3, insight_text, ha='center', va='bottom', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=3, pad=0.8),
            family='monospace')

    plt.tight_layout()
    plt.savefig('mindmaps/03_unified_view.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/03_unified_view.png")


def create_architecture_evolution():
    """Show evolution of neural architectures leading to diffusion."""
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Neural Architecture Evolution: From MLP to Diffusion', fontsize=28, fontweight='bold', pad=30)

    # Timeline of architectures - SCALED UP
    architectures = [
        ('MLP', '1980s', 2, '#e0e0e0'),
        ('CNN\n(LeNet)', '1998', 5.5, '#cce5ff'),
        ('ResNet', '2015', 9, '#d4edda'),
        ('Transformer', '2017', 12.5, '#fff3cd'),
        ('Neural ODE', '2018', 16, '#e2d5f1'),
        ('Diffusion', '2020', 19.5, '#ff6b6b'),
    ]

    # Key insights
    insights = [
        ('Basic\nfunction\napprox', 2, 3.5),
        ('Local\nconnectivity\nweight sharing', 5.5, 3.5),
        ('Skip\nconnections\ngradient flow', 9, 3.5),
        ('Self-attention\ndynamic\nweights', 12.5, 3.5),
        ('Continuous\ndepth\nODE solver', 16, 3.5),
        ('Noise +\nscore\nlearning', 19.5, 3.5),
    ]

    # Draw timeline - THICKER
    ax.plot([1, 21], [8, 8], 'k-', lw=4)
    for i, (name, year, x, color) in enumerate(architectures):
        # Circle on timeline - LARGER
        circle = plt.Circle((x, 8), 0.4, color='black', zorder=5)
        ax.add_patch(circle)
        ax.text(x, 9, year, ha='center', fontsize=14, fontweight='bold')

        # Architecture box - LARGER
        box = FancyBboxPatch((x-1.5, 5.8), 3, 1.8,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=3)
        ax.add_patch(box)
        ax.text(x, 6.7, name, ha='center', va='center', fontsize=15, fontweight='bold')

    # Draw insight boxes - LARGER
    for text, x, y in insights:
        box = FancyBboxPatch((x-1.3, y-1.1), 2.6, 2.2,
                              boxstyle="round,pad=0.03,rounding_size=0.1",
                              facecolor='white', edgecolor='gray', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=12)
        # Arrow from architecture to insight
        ax.annotate('', xy=(x, y+1.1), xytext=(x, 5.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Evolution arrows - THICKER
    for i in range(len(architectures)-1):
        x1 = architectures[i][2] + 1.5
        x2 = architectures[i+1][2] - 1.5
        ax.annotate('', xy=(x2, 6.7), xytext=(x1, 6.7),
                    arrowprops=dict(arrowstyle='->', color='green', lw=3))

    # Add evolution labels - LARGER
    evolution_labels = [
        ('+conv', 3.75),
        ('+skip', 7.25),
        ('+attention', 10.75),
        ('+continuous', 14.25),
        ('+stochastic', 17.75),
    ]
    for label, x in evolution_labels:
        ax.text(x, 7.2, label, ha='center', fontsize=13, color='green', fontweight='bold')

    # Bottom summary - LARGER
    ax.text(11, 1, 'Each architecture builds on the previous:\nMLP → add structure → add highways → add dynamics → add noise',
            ha='center', fontsize=16, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.8))

    plt.tight_layout()
    plt.savefig('mindmaps/04_architecture_evolution.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/04_architecture_evolution.png")


def create_loss_functions():
    """Show how loss functions derive from information theory."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    ax.set_title('Loss Functions and Their Information-Theoretic Roots', fontsize=28, fontweight='bold', pad=30)

    # Information theory core (top) - LARGER
    info_concepts = {
        'Entropy\nH(X)': (5, 11, '#cce5ff'),
        'KL Divergence\nD_KL(P||Q)': (10, 11, '#d4edda'),
        'Mutual Info\nI(X;Y)': (15, 11, '#fff3cd'),
    }

    # Loss functions (middle)
    losses = {
        'Cross-Entropy\nLoss': (5, 6.5, '#e6f2ff'),
        'ELBO\n(VAE Loss)': (10, 6.5, '#e6ffe6'),
        'Score Matching\nLoss': (15, 6.5, '#fff9e6'),
    }

    # Models (bottom)
    models = {
        'Classification\n(Softmax)': (5, 2, '#f0f0f0'),
        'VAE / β-VAE': (10, 2, '#f0f0f0'),
        'Diffusion\nModels': (15, 2, '#ffcccc'),
    }

    # Draw boxes - LARGER
    for group, y_base in [(info_concepts, 11), (losses, 6.5), (models, 2)]:
        for name, (x, y, color) in group.items():
            box = FancyBboxPatch((x-2, y-1), 4, 2,
                                  boxstyle="round,pad=0.05,rounding_size=0.2",
                                  facecolor=color, edgecolor='black', linewidth=3)
            ax.add_patch(box)
            ax.text(x, y, name, ha='center', va='center', fontsize=15, fontweight='bold')

    # Arrows with labels - LARGER
    arrows = [
        # Entropy to losses
        (5, 10, 5, 7.5, 'defines'),
        (5, 10, 10, 7.5, 'H(P,Q)=H+D_KL'),

        # KL to losses
        (10, 10, 5, 7.5, '+'),
        (10, 10, 10, 7.5, 'regularization'),
        (10, 10, 15, 7.5, 'equivalent'),

        # Mutual info
        (15, 10, 10, 7.5, 'bottleneck'),

        # Losses to models
        (5, 5.5, 5, 3, ''),
        (10, 5.5, 10, 3, ''),
        (15, 5.5, 15, 3, ''),
    ]

    for x1, y1, x2, y2, label in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        if label:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x+0.4, mid_y, label, fontsize=11, color='blue', rotation=0)

    # Section labels - LARGER
    ax.text(1, 11, 'INFORMATION\nTHEORY', ha='center', fontsize=14, fontweight='bold', color='navy')
    ax.text(1, 6.5, 'LOSS\nFUNCTIONS', ha='center', fontsize=14, fontweight='bold', color='darkgreen')
    ax.text(1, 2, 'MODEL\nTYPES', ha='center', fontsize=14, fontweight='bold', color='darkred')

    # Key formulas - LARGER
    formulas = """KEY FORMULAS:
─────────────────────────────────────────────
Entropy:        H(X) = -E[log p(X)]
Cross-Entropy:  H(P,Q) = -E_P[log Q]
KL Divergence:  D_KL(P||Q) = E_P[log P/Q]

Cross-Entropy = Entropy + KL Divergence
H(P,Q) = H(P) + D_KL(P||Q)

=> Minimizing CE ≈ Minimizing KL!"""
    ax.text(10, -0.8, formulas, ha='center', va='top', fontsize=13,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2, pad=0.8))

    plt.tight_layout()
    plt.savefig('mindmaps/05_loss_functions.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/05_loss_functions.png")


def create_svd_to_diffusion():
    """Show the core mathematical thread from SVD to Diffusion."""
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('The Core Thread: SVD → Whitening → VAE → Diffusion', fontsize=28, fontweight='bold', pad=30)

    # Main pipeline - LARGER
    pipeline = [
        ('Raw Data\nX', 2.5, 11, '#e0e0e0'),
        ('SVD\nX=UΣV\'', 6.5, 11, '#cce5ff'),
        ('Whitened\nData', 10.5, 11, '#d4edda'),
        ('Latent\nSpace z', 14.5, 11, '#fff3cd'),
        ('Generated\nSamples', 18.5, 11, '#ff6b6b'),
    ]

    # Draw pipeline boxes - LARGER
    for name, x, y, color in pipeline:
        box = FancyBboxPatch((x-1.6, y-1), 3.2, 2,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=3)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=15, fontweight='bold')

    # Arrows between pipeline stages - THICKER
    for i in range(len(pipeline)-1):
        x1 = pipeline[i][1] + 1.6
        x2 = pipeline[i+1][1] - 1.6
        y = pipeline[i][2]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=3))

    # Process labels - LARGER
    process_labels = [
        ('decompose', 4.5, 12.2),
        ('normalize\nΣ⁻¹', 8.5, 12.2),
        ('encode/\ndecode', 12.5, 12.2),
        ('sample +\ndenoise', 16.5, 12.2),
    ]
    for label, x, y in process_labels:
        ax.text(x, y, label, ha='center', fontsize=13, color='blue')

    # Methods below each stage - LARGER
    methods = [
        ('PCA\n(linear)', 6.5, 6, '#cce5ff'),
        ('Autoencoder\n(nonlinear)', 10.5, 6, '#d4edda'),
        ('VAE\n(probabilistic)', 14.5, 6, '#fff3cd'),
        ('Diffusion\n(iterative)', 18.5, 6, '#ff6b6b'),
    ]

    for name, x, y, color in methods:
        box = FancyBboxPatch((x-1.8, y-0.9), 3.6, 1.8,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=14)

    # Connect pipeline to methods
    connections = [
        (6.5, 10, 6.5, 6.9),  # SVD → PCA
        (10.5, 10, 10.5, 6.9),  # Whitened → AE
        (14.5, 10, 14.5, 6.9),  # Latent → VAE
        (18.5, 10, 18.5, 6.9),  # Generated → Diffusion
    ]
    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

    # Evolution arrows - THICKER
    for i in range(len(methods)-1):
        x1 = methods[i][1] + 1.8
        x2 = methods[i+1][1] - 1.8
        y = methods[i][2]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='green', lw=3))

    # Evolution labels - LARGER
    evo_labels = [
        ('+neural net', 8.5, 6.6),
        ('+KL term', 12.5, 6.6),
        ('+many steps', 16.5, 6.6),
    ]
    for label, x, y in evo_labels:
        ax.text(x, y, label, ha='center', fontsize=12, color='green', fontweight='bold')

    # Key insight - LARGER
    insight = """THE UNIFYING IDEA:

SVD whitens data → Covariance becomes Identity
VAE forces latent → N(0, I) via KL term
Diffusion endpoint → Pure Gaussian N(0, I)

All methods work with "whitened" representations!"""
    ax.text(11, 1.5, insight, ha='center', fontsize=15,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=3, pad=0.8),
            family='monospace')

    plt.tight_layout()
    plt.savefig('mindmaps/06_svd_to_diffusion.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/06_svd_to_diffusion.png")


def create_why_chain():
    """Create the 'Why' chain showing logical dependencies."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.axis('off')
    ax.set_title('The "Why" Chain: Tracing Concepts Backward', fontsize=28, fontweight='bold', pad=30)

    # Questions and answers (from top to bottom) - LARGER
    qa_pairs = [
        ('Q: Why do diffusion models work?', 'A: They learn the score function ∇log p(x,t)', '#ff6b6b'),
        ('Q: Why does learning the score help?', 'A: Langevin dynamics samples using only the score', '#ffd700'),
        ('Q: Why does Langevin work?', 'A: SDEs converge to stationary distribution', '#fff3cd'),
        ('Q: Why do we need SDEs?', 'A: Deterministic ODEs can\'t generate diversity', '#e2d5f1'),
        ('Q: Why use neural networks?', 'A: True score is intractable for complex data', '#d4edda'),
        ('Q: Why can NNs approximate functions?', 'A: Universal approximation + optimization', '#cce5ff'),
        ('Q: Why does backprop work?', 'A: Chain rule (automatic differentiation)', '#a8d5ff'),
        ('Q: Why do we need linear algebra?', 'A: NNs are matrix ops + nonlinearities', '#e0e0e0'),
    ]

    y_start = 16.5
    y_step = 2

    for i, (q, a, color) in enumerate(qa_pairs):
        y = y_start - i * y_step

        # Question box - LARGER
        q_box = FancyBboxPatch((1.5, y-0.5), 6.5, 1,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor=color, edgecolor='black', linewidth=3)
        ax.add_patch(q_box)
        ax.text(4.75, y, q, ha='center', va='center', fontsize=13, fontweight='bold')

        # Answer box - LARGER
        a_box = FancyBboxPatch((8.5, y-0.5), 7.5, 1,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor='white', edgecolor=color, linewidth=3)
        ax.add_patch(a_box)
        ax.text(12.25, y, a, ha='center', va='center', fontsize=12)

        # Arrow from answer to next question
        if i < len(qa_pairs) - 1:
            ax.annotate('', xy=(4.75, y - 0.7), xytext=(12.25, y - 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2,
                                        connectionstyle='arc3,rad=-0.3'))

    # Chapter labels on the right - LARGER
    chapters = [
        ('Ch.9', 16.5), ('Ch.7', 14.5), ('Ch.7', 12.5),
        ('Ch.2', 10.5), ('Ch.4', 8.5), ('Ch.4', 6.5),
        ('Ch.2', 4.5), ('Ch.1', 2.5),
    ]
    for ch, y in chapters:
        ax.text(17, y, ch, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='lightgray', pad=0.5))

    # Add "READ THIS WAY" arrow - LARGER
    ax.text(0.7, 5, 'READ\nBOTTOM\nUP', fontsize=16, fontweight='bold',
            color='red', ha='center')
    ax.annotate('', xy=(0.7, 16), xytext=(0.7, 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=4))

    plt.tight_layout()
    plt.savefig('mindmaps/07_why_chain.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: mindmaps/07_why_chain.png")


def main():
    """Generate all mind maps."""
    import os

    # Create output directory
    os.makedirs('mindmaps', exist_ok=True)

    print("Generating mind maps...")
    print("=" * 50)

    create_master_mindmap()
    create_concept_flow()
    create_unified_view()
    create_architecture_evolution()
    create_loss_functions()
    create_svd_to_diffusion()
    create_why_chain()

    print("=" * 50)
    print("\nAll mind maps saved to mindmaps/ folder!")
    print("\nFiles created:")
    for f in sorted(os.listdir('mindmaps')):
        if f.endswith('.png'):
            print(f"  - mindmaps/{f}")


if __name__ == '__main__':
    main()
