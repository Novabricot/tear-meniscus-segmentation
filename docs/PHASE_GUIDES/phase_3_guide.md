# PHASE 3: Advanced Learning Techniques Implementation Guide

**Duration**: Weeks 5-9  
**Target F1**: 0.94-0.96  
**Effort Level**: High  
**GPU Required**: 10-16 GB  

## Overview

Phase 3 introduces advanced techniques to squeeze additional performance gains:
- **3A**: Self-supervised learning (SimCLR pretraining)
- **3B**: Domain adversarial training (ring robustness)
- **3C**: Ensemble methods (stability and uncertainty)

These can be run in parallel (3B and 3C while 3A pretrains).

---

## Phase 3A: Self-Supervised Pretraining (SimCLR)

### Motivation
- Dataset has 1,739 unlabeled IR images
- These contain valuable tear meniscus information
- Can use them to improve encoder representations

### How SimCLR Works
```
1. Take one image → create 2 different random augmentations
2. Pass both through encoder → get embeddings z1, z2
3. Loss = maximize similarity of (z1, z2) for same image
         minimize similarity for different images
4. Train without labels!
5. Transfer encoder to segmentation task
```

### Implementation Timeline (7-10 days)

- [ ] **Day 1-2: Setup**
  - [ ] Install `lightly` library: `pip install lightly`
  - [ ] Prepare all 3,432 images (color + IR, no labels)
  - [ ] Create augmentation for contrastive learning

- [ ] **Day 3-7: Pretraining**
  - [ ] Implement SimCLR in `PHASE_3_ADVANCED/scripts/train_simclr_pretrain.py`
  - [ ] Train for 100 epochs
  - [ ] Monitor loss convergence
  - [ ] Save pretrained encoder

- [ ] **Day 8-10: Transfer Learning**
  - [ ] Load pretrained encoder
  - [ ] Attach segmentation head
  - [ ] Fine-tune on labeled tear meniscus data only
  - [ ] Compare with Phase 2 baseline

### Key Code Snippets

```python
# PHASE_3_ADVANCED/scripts/train_simclr_pretrain.py
from lightly.models import SimCLR
from lightly.loss import NTXentLoss

backbone = torchvision.models.resnet50()
model = SimCLR(backbone, hidden_dim=2048, out_dim=128)

# Contrastive loss
loss_fn = NTXentLoss(temperature=0.07)

# Training loop (no labels needed!)
for epoch in range(100):
    for batch in all_images_dataloader:  # ALL images
        x0, x1 = batch  # Two augmentations of same image
        
        z0 = model(x0)
        z1 = model(x1)
        
        loss = loss_fn(z0, z1)
        loss.backward()
        optimizer.step()

# Save encoder
torch.save(model.backbone.state_dict(), 
           'PHASE_3_ADVANCED/models/simclr_encoder.pth')
```

### Expected Impact
- **Baseline (Phase 2)**: F1 = 0.9287
- **With SimCLR**: F1 = 0.9487 (+2% improvement)
- **Why**: Better feature representations from contrastive learning

---

## Phase 3B: Domain Adversarial Training

### Motivation
- Current model may learn Placido ring patterns (dataset artifacts)
- For future headset deployment: no rings present
- Need ring-invariant features

### How Domain Adversarial Training Works
```
1. Segmentation task (normal):
   Image → Model → Mask prediction
   
2. Domain discrimination (adversarial):
   Features → Domain discriminator → "Has rings?" prediction
   
3. Adversarial loss:
   Loss = SegmentationLoss - λ × DomainAdversarialLoss
   Goal: Fool discriminator (features agnostic to rings)
```

### Implementation Timeline (3-5 days)

- [ ] **Day 1: Data Preparation**
  - [ ] Create synthetic "no-ring" variants
  - [ ] Mask central 200px radius (where Placido rings are)
  - [ ] Label: original=1 (has rings), masked=0 (no rings)

- [ ] **Day 2-4: Training**
  - [ ] Implement domain discriminator
  - [ ] Add adversarial loss weighting
  - [ ] Train SegFormer-B3 with domain loss
  - [ ] Monitor segmentation AND domain accuracy

- [ ] **Day 5: Evaluation**
  - [ ] Test on original (with rings)
  - [ ] Test on masked (without rings)
  - [ ] Compare F1 drop (small drop = good robustness)

### Key Code Snippets

```python
# PHASE_3_ADVANCED/scripts/train_domain_adversarial.py
class DomainAdversarialSegmentation(nn.Module):
    def __init__(self, segmentation_model):
        super().__init__()
        self.encoder = segmentation_model.encoder
        self.seg_head = segmentation_model.decoder + segmentation_model.head
        
        # Domain discriminator (small network)
        self.domain_head = nn.Sequential(
            nn.Linear(256, 128),  # Features from encoder
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary: has rings or not
        )
    
    def forward(self, x, return_domain=False):
        features = self.encoder(x)
        seg_output = self.seg_head(features)
        
        if return_domain:
            domain_output = self.domain_head(features.mean(dim=[2, 3]))
            return seg_output, domain_output
        return seg_output

# Training with adversarial loss
model = DomainAdversarialSegmentation(segformer_model)
lambda_adv = 0.1  # Adversarial loss weight

for batch in dataloader:
    seg_pred, domain_pred = model(batch['image'], return_domain=True)
    
    # Segmentation loss (normal)
    seg_loss = dice_loss(seg_pred, batch['mask'])
    
    # Domain loss (adversarial - fool discriminator)
    domain_loss = ce_loss(domain_pred, batch['domain_label'])
    
    # Total loss: maximize segmentation, minimize domain discrimination
    total_loss = seg_loss - lambda_adv * domain_loss
    total_loss.backward()
```

### Expected Results
| Test Set | Phase 2 | Phase 3B | Drop |
|---|---|---|---|
| Original (rings) | 0.9287 | 0.9287 | 0% |
| Masked (no rings) | 0.8234 | 0.8934 | 1% |
| **Robustness** | Poor | **Good** | ↓ |

### Ring Removal Strategy

```python
def remove_rings_synthetic(image, radius=200):
    """Mask central region (where Placido rings are)"""
    H, W = image.shape[:2]
    center = (H // 2, W // 2)
    
    # Create circular mask
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    
    # Inpaint masked region (Gaussian interpolation)
    inpainted = cv2.inpaint(
        image.astype(np.uint8),
        (1 - mask) * 255,  # Invert mask
        3,
        cv2.INPAINT_TELEA
    )
    
    return inpainted
```

---

## Phase 3C: Ensemble Methods

### Motivation
- Single models have inherent uncertainty
- Multiple models with different random seeds = better generalization
- Ensemble reduces variance, improves robustness

### How Ensemble Works
```
Model 1 (SegFormer-B3, seed=0) → Prediction 1
Model 2 (SegFormer-B3, seed=1) → Prediction 2
Model 3 (SegFormer-B3, seed=2) → Prediction 3
Model 4 (SegFormer-B5, seed=0) → Prediction 4

Average ensemble:
Final prediction = (P1 + P2 + P3 + P4) / 4

Uncertainty:
Variance = Var(P1, P2, P3, P4)  ← confidence score
```

### Implementation Timeline (3-4 days, parallel with 3B)

- [ ] **Day 1-2: Train Multiple Seeds**
  - [ ] Retrain SegFormer-B3 with seeds: 0, 1, 2
  - [ ] Each model → different random initialization
  - [ ] Best validation checkpoint for each

- [ ] **Day 3: Ensemble Integration**
  - [ ] Load all 4 models (3× B3 + 1× B5 from Phase 2)
  - [ ] Implement ensemble averaging
  - [ ] Compute uncertainty per pixel

- [ ] **Day 4: Evaluation**
  - [ ] Compare individual vs ensemble F1
  - [ ] Quantify uncertainty estimates
  - [ ] Identify high-confidence regions

### Key Code Snippets

```python
# PHASE_3_ADVANCED/scripts/ensemble_inference.py
class EnsembleSegmentation:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = load_model(path)
            model.eval()
            self.models.append(model)
        self.device = next(self.models[0].parameters()).device
    
    def predict_with_uncertainty(self, image):
        """Get ensemble prediction and uncertainty"""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                image_t = image.to(self.device)
                logits = model(image_t)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack: (num_models, 1, H, W, classes)
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty (variance)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty
    
    def evaluate_ensemble(self, test_loader):
        """Evaluate ensemble on test set"""
        all_f1 = []
        all_uncertainty = []
        
        for batch in test_loader:
            pred, uncert = self.predict_with_uncertainty(batch['image'])
            
            # Calculate F1
            f1 = calculate_f1(pred, batch['mask'])
            all_f1.append(f1)
            all_uncertainty.append(uncert.mean().item())
        
        avg_f1 = np.mean(all_f1)
        avg_uncertainty = np.mean(all_uncertainty)
        
        return {
            'f1': avg_f1,
            'uncertainty': avg_uncertainty
        }

# Usage
ensemble = EnsembleSegmentation([
    'models/phase_2/segformer_b3_seed0.pth',
    'models/phase_2/segformer_b3_seed1.pth',
    'models/phase_2/segformer_b3_seed2.pth',
    'models/phase_2/segformer_b5.pth',
])

pred, uncertainty = ensemble.predict_with_uncertainty(test_image)
print(f"Ensemble F1: {pred}")
print(f"Uncertainty: {uncertainty.mean():.4f}")
```

### Expected Results
| Model | F1 | Std Dev | Stability |
|---|---|---|---|
| Single Model | 0.9287 | 0.0234 | Good |
| Ensemble (4 models) | 0.9387 | 0.0087 | **Better** |

*Ensemble slightly higher F1 + much lower variance = better for production*

---

## Phase 3 Timeline (Flexible, Can Parallelize)

```
Week 5:
├─ 3A: Days 1-5  → SimCLR pretraining (main focus)
└─ 3B: Days 1-2  → Prepare domain data

Week 6:
├─ 3A: Finish transfer learning
├─ 3B: Days 1-3  → Domain adversarial training
└─ 3C: Days 1-2  → Retrain models with different seeds

Week 7:
├─ 3B: Finish evaluation
├─ 3C: Days 1-3  → Ensemble training & evaluation
└─ All: Compare results

Week 8-9:
├─ Ablation studies (which technique helps most?)
├─ Final evaluation on full test set
├─ Write documentation
└─ Prepare paper/presentation
```

---

## Expected Final Results

### Performance Progression

| Phase | Technique | F1 Score | Improvement |
|---|---|---|---|
| Baseline | U-Net | 0.9220 | — |
| **1** | DeepLabV3+ ResNet-101 | 0.9087 | -0.13% (maintenance) |
| **2** | SegFormer-B3 | 0.9287 | +0.67% |
| **3A** | + SimCLR pretraining | 0.9487 | +2.67% |
| **3B** | + Domain adversarial | 0.9487 | +2.67% (robustness) |
| **3C** | + Ensemble (4 models) | 0.9487 | +2.67% |

**Final Target**: F1 ≥ 0.94-0.96 ✅

### Robustness Gains (3B)
| Test Condition | Phase 2 | Phase 3B |
|---|---|---|
| Original images (with rings) | 0.9287 | 0.9287 |
| Ring-masked images | 0.8234 | 0.8934 |
| **Ring-invariance gain** | 10.53% drop | **3.80% drop** |

### Uncertainty Improvements (3C)
| Metric | Individual Model | Ensemble |
|---|---|---|
| Mean F1 | 0.9287 | 0.9387 |
| Std Dev | 0.0234 | 0.0087 |
| Confidence | ±2.34% | **±0.87%** |

---

## Advanced Configuration Tuning

### SimCLR Temperature
```yaml
# PHASE_3_ADVANCED/configs/phase3_config.yaml
simclr_pretraining:
  training:
    temperature: 0.07  # Lower = sharper contrast
    # Default is 0.07 (good for medical imaging)
    # Higher (0.5) = softer, may overfit
    # Lower (0.02) = too sharp, slow convergence
```

### Domain Adversarial Loss Weight
```yaml
domain_adversarial:
  training:
    lambda_adversarial: 0.1  # Adversarial loss weight
    # 0.01 = weak domain adaptation
    # 0.1 = balanced
    # 1.0 = strong (may hurt segmentation)
```

### Ensemble Voting Strategies
```python
# Average ensemble (recommended)
ensemble_pred = predictions.mean(dim=0)

# Weighted ensemble (by validation F1)
weights = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Normalized
ensemble_pred = (predictions * weights.view(-1, 1, 1, 1)).sum(dim=0)

# Majority voting (only for classification)
classes = torch.argmax(predictions, dim=-1)  # (N, H, W)
ensemble_class = torch.mode(classes, dim=0)[0]  # Most common class
```

---

## Troubleshooting Phase 3

### SimCLR Pretraining
**Issue**: Loss doesn't decrease, stays high
- **Cause**: Learning rate too high or too low
- **Solution**: Try LR = 0.003, 0.001, 0.0003
- **Check**: Similarity between z1 and z2 should increase each epoch

**Issue**: Out of memory during pretraining
- **Cause**: Batch size 64 × 2 augmentations = 128 effective
- **Solution**: Reduce to batch 32 or 16

### Domain Adversarial Training
**Issue**: Domain discriminator accuracy too high (>95%)
- **Cause**: Too easy to distinguish rings from no rings
- **Solution**: Increase `lambda_adversarial` from 0.1 to 0.3-0.5

**Issue**: Segmentation accuracy drops with domain loss
- **Cause**: Adversarial loss weight too high
- **Solution**: Decrease `lambda_adversarial` to 0.01-0.05

### Ensemble Methods
**Issue**: Ensemble F1 lower than best individual model
- **Cause**: Including weak models in ensemble
- **Solution**: Only use models with F1 > 0.92
- **Alternative**: Use weighted ensemble (weight by validation F1)

---

## Documentation & Reporting

### Create Phase 3 Summary

```markdown
# Phase 3 Results Summary

## 3A: Self-Supervised Pretraining
- Pretrained on: 3,432 images (labeled + unlabeled IR)
- Epochs: 100
- Improvement: F1 0.9287 → 0.9487 (+2%)
- Key learning: Unlabeled IR images help significantly

## 3B: Domain Adversarial Training
- Ring masking radius: 200px
- Lambda adversarial: 0.1
- Ring-invariance improvement: 10.53% → 3.80% drop
- Conclusion: Model more robust to domain shift

## 3C: Ensemble Methods
- Models: SegFormer-B3 (3 seeds) + B5 (best)
- Ensemble strategy: Average pooling
- F1: 0.9287 → 0.9387
- Uncertainty reduction: ±2.34% → ±0.87%
- Conclusion: More stable for production deployment
```

### Final Comparison Plot

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: F1 progression
phases = ['Baseline\nU-Net', 'Phase 1\nResNet-101', 'Phase 2\nSegFormer-B3', 
          'Phase 3A\n+ SimCLR', 'Phase 3B\n+ Domain Adv', 'Phase 3C\n+ Ensemble']
f1_scores = [0.9220, 0.9087, 0.9287, 0.9487, 0.9487, 0.9387]
axes[0,0].bar(phases, f1_scores, color=['gray', 'blue', 'green', 'orange', 'red', 'purple'])
axes[0,0].set_ylabel('F1 Score')
axes[0,0].set_title('F1 Score Progression')
axes[0,0].set_ylim([0.90, 0.95])

# Plot 2: Ring robustness
models = ['Phase 2', 'Phase 3B']
original = [0.9287, 0.9287]
masked = [0.8234, 0.8934]
x = np.arange(len(models))
axes[0,1].bar(x - 0.2, original, 0.4, label='Original (rings)')
axes[0,1].bar(x + 0.2, masked, 0.4, label='Masked (no rings)')
axes[0,1].set_ylabel('F1 Score')
axes[0,1].set_title('Domain Robustness')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(models)
axes[0,1].legend()

# Plot 3: Uncertainty reduction
models = ['Individual\nModels', 'Ensemble\n(4 models)']
uncertainty = [0.0234, 0.0087]
axes[1,0].bar(models, uncertainty, color=['blue', 'green'])
axes[1,0].set_ylabel('Std Dev of F1')
axes[1,0].set_title('Prediction Uncertainty')

# Plot 4: Summary table
axes[1,1].axis('tight')
axes[1,1].axis('off')
table_data = [
    ['Phase', 'Technique', 'F1', 'Improvement'],
    ['1', 'DeepLabV3+ ResNet-101', '0.9087', '-0.13%'],
    ['2', 'SegFormer-B3', '0.9287', '+0.67%'],
    ['3A', '+ SimCLR', '0.9487', '+2.67%'],
    ['3B', '+ Domain Adv', '0.9487', '+2.67%*'],
    ['3C', '+ Ensemble', '0.9387†', '+2.67%'],
]
table = axes[1,1].table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)

plt.tight_layout()
plt.savefig('docs/phase_3_summary.png', dpi=150, bbox_inches='tight')
```

---

## Next Steps After Phase 3

1. **Finalization**:
   - Select best model (likely 3C ensemble or 3A-pretrained)
   - Create inference-optimized version
   - Document all hyperparameters

2. **Testing & Validation**:
   - Cross-validation on full dataset
   - External validation (if possible)
   - Robustness testing (augmentation, noise)

3. **Publication**:
   - Prepare paper/technical report
   - Include comparison tables and plots
   - Document methods and results

4. **Repository**:
   - Push final code to GitHub
   - Create RESULTS.md with findings
   - Tag release (v1.0)

---

**Status**: Ready for implementation  
**Previous Phase**: Phase 2 (complete)  
**Final Timeline**: Weeks 5-9 (5 weeks total)  
**Target Completion**: Week 9 of project
