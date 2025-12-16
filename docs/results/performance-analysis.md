# Performance Analysis

Comparative results between Bahdanau and Luong attention mechanisms.

## Training Dynamics

### Loss Curves (Typical Results)

| Epoch | Bahdanau Loss | Luong Loss |
|-------|---------------|------------|
| 1 | ~4.2 | ~4.1 |
| 2 | ~3.5 | ~3.4 |
| 3 | ~2.9 | ~2.8 |
| 5 | ~2.1 | ~2.0 |
| 10 | ~1.3 | ~1.3 |

*Note: Exact values vary with random seeds and hardware.*

## Overall Comparison

| Metric | Bahdanau | Luong (General) |
|--------|----------|-----------------|
| Final Loss | ~1.2-1.4 | ~1.2-1.4 |
| Convergence Speed | Slightly slower | Slightly faster |
| Parameters | More | Fewer |
| Compute/Step | Higher | Lower |
| Alignment Quality | Excellent | Excellent |

## Key Observations

### 1. Similar Final Performance
Both mechanisms achieve comparable final loss values. The difference is typically < 0.1 loss points.

### 2. Luong Converges Faster
The simpler multiplicative scoring leads to slightly faster initial convergence.

### 3. Bahdanau More Expressive
The additive mechanism with tanh provides more nuanced alignment, especially visible in attention visualizations.

### 4. Computational Efficiency
- **Luong (dot)**: Fastest, no learned parameters in scoring
- **Luong (general)**: Middle ground
- **Bahdanau**: Most expressive but slowest

## When to Choose Each

| Scenario | Recommended |
|----------|-------------|
| Speed-critical deployment | Luong (dot) |
| Research/Experimentation | Bahdanau |
| Transformer integration | Scaled Dot-Product |
| Dimension mismatch | Luong (general) or Bahdanau |

## Theoretical vs Practical

| Claim (Literature) | Our Observation |
|--------------------|-----------------|
| "Bahdanau is more flexible" | ✓ Confirmed |
| "Luong is faster" | ✓ Confirmed |
| "Performance is similar" | ✓ Confirmed |
| "Attention helps long sequences" | ✓ Confirmed |

---

[← Experimental Setup](experimental-setup.md) | [Next: Attention Visualizations →](attention-visualizations.md)
