# Research Findings - Tear Meniscus Segmentation Improvement Project

**Project**: Improving tear meniscus semantic segmentation for dry eye disease diagnosis  
**Timeline**: May 2026 - September 2026  
**Status**: Active development  

## Overview

This document collects key findings, insights, and lessons learned throughout the research project.

---

## Phase Insights

### Phase 1: Quick Wins (DeepLabV3+ ResNet-101)

**Key Findings**:
- [To be populated during Phase 1]

**Lessons Learned**:
- [TBD]

**Best Practices Identified**:
- [TBD]

---

### Phase 2: Modern Architectures (SegFormer)

**Key Findings**:
- [To be populated during Phase 2]

**Transformer vs CNN Comparison**:
- [TBD]

**Inference Speed Trade-offs**:
- [TBD]

---

### Phase 3A: Self-Supervised Learning (SimCLR)

**Impact of Unlabeled Data**:
- [To be populated during Phase 3A]
- IR images: [number] unlabeled images
- Improvement from pretraining: [TBD]

**Contrastive Learning Insights**:
- [TBD]

---

### Phase 3B: Domain Adversarial Training

**Ring Robustness**:
- [To be populated during Phase 3B]
- Ring removal effectiveness: [TBD]
- Domain shift reduction: [TBD]

**Headset Deployment Implications**:
- [TBD]

---

### Phase 3C: Ensemble Methods

**Ensemble Gains**:
- [To be populated during Phase 3C]
- F1 improvement: [TBD]
- Uncertainty reduction: [TBD]

**Confidence Calibration**:
- [TBD]

---

## Dataset Observations

### Image Quality
- [To be recorded]

### Placido Ring Patterns
- Present in all color images
- Diameter range: [TBD]
- Potential as feature or confound: [TBD]

### Iris Color Distribution
- [To be analyzed]
- Diversity across samples: [TBD]

### Tear Meniscus Characteristics
- [To be analyzed]

---

## Technical Insights

### Augmentation Effectiveness

**What Worked**:
- [TBD]

**What Didn't Work**:
- [TBD]

**Surprising Findings**:
- [TBD]

### Learning Dynamics

**Convergence Patterns**:
- [TBD]

**Overfitting Behavior**:
- [TBD]

**Learning Rate Sensitivity**:
- [TBD]

---

## Baseline Comparisons

### U-Net (Published Paper)
- F1: 0.9220
- Recall: 0.9127
- Precision: 0.9367
- MIoU: 0.9279

### Phase 1 Results (Expected)
- [TBD]

### Phase 2 Results (Expected)
- [TBD]

### Phase 3 Results (Expected)
- [TBD]

---

## Computational Efficiency

### Training Time Analysis
- [TBD]

### Inference Speed
- [TBD]

### Memory Usage
- [TBD]

### GPU Utilization
- [TBD]

---

## Generalization & Robustness

### Cross-Domain Performance
- Original images with rings: [TBD]
- Ring-masked synthetic: [TBD]
- Potential headset images: [TBD]

### Uncertainty Calibration
- Prediction confidence: [TBD]
- Correctness correlation: [TBD]

### Failure Modes
- [TBD]

---

## Clinical Implications

### Tear Meniscus Measurement Accuracy
- [To be analyzed]

### Dry Eye Detection Capability
- [TBD]

### Integration with Clinical Workflow
- [TBD]

---

## Published Paper Findings

### Original Dataset Paper
- Citation: [Figshare Link]
- Key baselines: U-Net (0.9220), ResUNet (0.9025), DeepLabV3+ (0.8705)
- Data source: 5 clinical centers in China
- Device: Oculus Keratograph 5M

### Reproducibility Notes
- [TBD] - Are our results matching published baselines?
- [TBD] - What differences account for variations?

---

## Recommendations for Future Work

### Short-term (Next Phase)
- [TBD]

### Medium-term (After Phase 3)
- [TBD]

### Long-term (Production Deployment)
- [TBD]

---

## Code & Implementation Insights

### Framework Choices
- PyTorch vs TensorFlow: PyTorch chosen for flexibility
- Segmentation models library: Good for architecture exploration
- Transformers library: Standard for SegFormer

### Development Tools
- [TBD]

### Reproducibility
- Random seeds: Fixed at 42
- Deterministic mode: Enabled
- Version control: Git with clear commit messages

---

## Paper-Relevant Findings

### Novelty Contributions
1. [TBD]
2. [TBD]
3. [TBD]

### Experimental Design
- [TBD]

### Statistical Significance
- [TBD]

### Comparison with Related Work
- [TBD]

---

## Limitations & Caveats

### Dataset Limitations
- Small size (3,432 images)
- Single imaging device (Keratograph 5M)
- Limited demographic diversity (brown/grey eyes)
- No explicit dry eye diagnosis labels

### Methodology Limitations
- [TBD]

### Generalization Concerns
- [TBD]

---

## Interesting Observations

### Unexpected Findings
- [TBD]

### Surprising Model Behaviors
- [TBD]

### Interesting Failure Cases
- [TBD]

---

## Author Notes

**Researcher**: Elora (@Novabricot)  
**Institution**: NAIST  
**Contact**: [TBD]  

---

**This document is updated throughout the project.**  
**Last Updated**: May 25, 2026  
**Next Update**: After Phase 1 completion (Week 2)
