# PHASE 1: Quick Wins Implementation Guide

**Duration**: Weeks 1-2  
**Target F1**: 0.90-0.91 (+2-3% improvement)  
**Effort Level**: Low  
**GPU Required**: 6-8 GB  

## Overview

Phase 1 focuses on quick, high-impact improvements using an upgraded architecture and enhanced data augmentation without requiring complex techniques.

### Key Techniques
1. **DeepLabV3+ ResNet-101** (upgrade from ResNet-50)
2. **Enhanced augmentation pipeline** (ElasticTransform, GridDistortion, Noise, MixUp)
3. **Progressive augmentation** (increase intensity during training)
4. **Mixed precision training** (for faster convergence)

---

## Baseline Comparison

| Model | F1 Score | Improvement |
|---|---|---|
| U-Net (baseline) | 0.9220 | — |
| **Phase 1 Target** | **0.91** | **+0.0% (maintenance)** |
| **Phase 1 Expected** | **0.94** | **+1-2%** |

*Note: Conservative target is 0.90; expect to reach ~0.91-0.93 with full implementation*

---

## Setup Checklist

### Week 1: Initial Setup
- [ ] **Day 1-2: Environment Setup**
  - [ ] Create Python virtual environment
  - [ ] Install dependencies from `requirements.txt`
  - [ ] Verify PyTorch and GPU availability
  - [ ] Download Figshare dataset (~11 GB)

- [ ] **Day 2-3: Data Preparation**
  - [ ] Extract dataset to `data/raw/`
  - [ ] Run `scripts/preprocess_data.py` to create train/val/test splits
  - [ ] Verify dataset structure in `data/processed/`
  - [ ] Create sample images for augmentation testing

- [ ] **Day 3-4: Code Development**
  - [ ] Implement `scripts/utils/data_loader.py` for dataset loading
  - [ ] Implement `scripts/utils/augmentation.py` with Phase 1 augmentations
  - [ ] Implement `scripts/utils/metrics.py` for F1/recall/precision/MIoU
  - [ ] Implement base training loop in `scripts/utils/trainer.py`

- [ ] **Day 5: Phase 1 Training Script**
  - [ ] Implement `PHASE_1_QUICKWINS/scripts/train_phase1.py`
  - [ ] Build DeepLabV3+ ResNet-101 model
  - [ ] Add loss functions (Dice, Cross-Entropy)
  - [ ] Setup TensorBoard logging

### Week 2: Training and Validation
- [ ] **Day 1-2: Training**
  - [ ] Start training with 100 epochs
  - [ ] Monitor loss curves on TensorBoard
  - [ ] Check GPU utilization and memory
  - [ ] Adjust batch size if OOM occurs

- [ ] **Day 3: Checkpoint Management**
  - [ ] Save best model based on validation F1
  - [ ] Save all 5-10 epoch checkpoints for analysis
  - [ ] Log training time per epoch

- [ ] **Day 4: Evaluation**
  - [ ] Run `scripts/eval.py` on test set
  - [ ] Generate metrics spreadsheet:
    - F1, Recall, Precision, MIoU
    - Training time
    - Inference speed (FPS)
  - [ ] Create loss curve plots

- [ ] **Day 5: Documentation**
  - [ ] Document findings in `docs/IMPLEMENTATION_NOTES.md`
  - [ ] Compare with baseline U-Net results
  - [ ] Identify what worked and what didn't
  - [ ] Plan improvements for Phase 2

---

## Detailed Implementation

### Step 1: Model Building

```python
# PHASE_1_QUICKWINS/scripts/train_phase1.py
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="resnet101",      # Larger backbone
    encoder_weights="imagenet",    # Pretrained weights
    in_channels=3,
    classes=2,                     # background + tear meniscus
    activation="sigmoid"
)

model = model.to(device)
```

**Configuration**: See `PHASE_1_QUICKWINS/configs/phase1_config.yaml`

### Step 2: Data Augmentation

```python
# scripts/utils/augmentation.py
import albumentations as A

train_transform = A.Compose([
    # Color/Intensity
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    A.CLAHE(p=0.2),
    
    # Geometric
    A.ElasticTransform(alpha=1, sigma=5, p=0.3),
    A.GridDistortion(p=0.2),
    
    # Domain
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.ISONoise(p=0.1),
    
    # Standard
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.2),
    
    # Mixup for segmentation
    # (Custom implementation)
])
```

### Step 3: Loss Function

```python
# Use Dice Loss (good for class imbalance)
from torch.nn import BCEWithLogitsLoss
import segmentation_models_pytorch as smp

loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

# Optional: combine with BCE
# loss_fn = smp.losses.JaccardLoss(mode='binary', from_logits=True)
```

### Step 4: Training Loop

```python
for epoch in range(num_epochs):
    # Update augmentation intensity (progressive)
    intensity = epoch / num_epochs
    train_loader.dataset.update_augmentation_intensity(intensity)
    
    # Train epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
    
    # Validate
    val_loss, val_f1, val_recall, val_precision, val_miou = validate(
        model, val_loader
    )
    
    # Log metrics
    logger.info(f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val MIoU: {val_miou:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'PHASE_1_QUICKWINS/models/best_model.pth')
```

### Step 5: Metrics Calculation

```python
# scripts/utils/metrics.py
from sklearn.metrics import f1_score, recall_score, precision_score

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """Calculate F1, Recall, Precision, MIoU"""
    
    # Binarize predictions
    pred_binary = (pred_mask > threshold).astype(int).flatten()
    true_binary = true_mask.astype(int).flatten()
    
    f1 = f1_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary)
    
    # MIoU (Intersection over Union)
    intersection = np.sum(pred_binary * true_binary)
    union = np.sum(pred_binary + true_binary - pred_binary * true_binary)
    miou = intersection / union if union > 0 else 0
    
    return {
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'miou': miou
    }
```

---

## Expected Results

### Success Criteria
- ✅ **F1 ≥ 0.90** on test set (minimum)
- ✅ **F1 ≥ 0.91** on test set (expected)
- ✅ Training converges smoothly (loss decreases)
- ✅ No out-of-memory errors
- ✅ Training completes in <20 hours

### Comparison with Baseline
| Metric | U-Net (Paper) | Phase 1 Expected |
|---|---|---|
| F1 | 0.9220 | 0.91-0.93 |
| Recall | 0.9127 | 0.92-0.93 |
| Precision | 0.9367 | 0.92-0.94 |
| MIoU | 0.9279 | 0.91-0.93 |

*Note: Small variations expected due to different train/val/test splits*

---

## Troubleshooting

### Issue: Out of Memory (OOM)
- **Solution**: Reduce batch size from 8 to 4 or 2
- **Alternative**: Use gradient accumulation
- **Code**:
```python
accumulation_steps = 2  # Effective batch = 8 with batch_size=4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = loss_fn(outputs, batch['mask'])
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Loss doesn't decrease
- **Possible causes**:
  - Learning rate too high or too low
  - Data loader returning wrong shapes
  - Loss function not suited for task
- **Solutions**:
  - Try LR = 0.0001 or 0.00001
  - Print batch shapes: `print(batch['image'].shape, batch['mask'].shape)`
  - Try BCE + Dice loss combination

### Issue: Low validation F1 despite low training loss (overfitting)
- **Solution**: Increase augmentation intensity
- **Alternative**: Add dropout or regularization
- **Code**:
```python
# Increase augmentation probabilities
brightness_limit=0.3  # was 0.2
contrast_limit=0.3    # was 0.2
p_geo=0.4             # geometric augmentation probability
```

### Issue: GPU utilization is low (<50%)
- **Possible cause**: Data loading bottleneck
- **Solution**: Increase `num_workers` in DataLoader
- **Code**:
```python
train_loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,  # Increase from 4
    pin_memory=True,
    prefetch_factor=2
)
```

---

## Monitoring Progress

### Key Metrics to Track (per epoch)
```csv
epoch,train_loss,val_loss,val_f1,val_recall,val_precision,val_miou,lr,time_sec
1,0.2543,0.2189,0.8654,0.8521,0.8798,0.8412,0.0010,45
2,0.2341,0.2087,0.8792,0.8654,0.8937,0.8543,0.0010,44
...
100,0.1234,0.1456,0.9087,0.9012,0.9167,0.8765,0.0001,41
```

Save to `PHASE_1_QUICKWINS/results/metrics.csv`

### Create Loss Curve Plot
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(metrics['train_loss'], label='Train')
plt.plot(metrics['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 3, 2)
plt.plot(metrics['val_f1'], label='F1')
plt.plot(metrics['val_miou'], label='MIoU')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Validation Metrics')

plt.savefig('PHASE_1_QUICKWINS/results/loss_curves.png', dpi=150, bbox_inches='tight')
```

---

## Next Steps (Week 3 onwards)

Once Phase 1 is complete:

1. **Document findings**:
   - What augmentations helped most?
   - What was the learning rate sweet spot?
   - How does ResNet-101 compare to ResNet-50?

2. **Prepare for Phase 2**:
   - Clean up best model checkpoint
   - Note any data preprocessing tricks
   - Prepare comparison baseline

3. **Commit to GitHub**:
   ```bash
   git add PHASE_1_QUICKWINS/ scripts/utils/ data/processed/
   git commit -m "feat: Phase 1 complete - DeepLabV3+ ResNet-101, F1=0.91+"
   git push
   ```

---

## Resources

### Key Papers
- DeepLabV3+: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- Advanced Augmentations: [Albumentations Library Documentation](https://albumentations.ai/)
- Mixed Precision Training: [PyTorch AMP Guide](https://pytorch.org/docs/stable/amp.html)

### Debugging Tools
- TensorBoard: `tensorboard --logdir PHASE_1_QUICKWINS/results/`
- Python Debugger: `import pdb; pdb.set_trace()`
- GPU Monitor: `nvidia-smi dmon` (real-time GPU utilization)

### Common Commands
```bash
# Monitor training in real-time
tensorboard --logdir PHASE_1_QUICKWINS/results/ --port=6006

# Check GPU memory
nvidia-smi

# Kill training if needed
pkill -f train_phase1.py

# Evaluate on test set
python scripts/eval.py --model PHASE_1_QUICKWINS/models/best_model.pth
```

---

**Status**: Ready for implementation  
**Last Updated**: May 25, 2026  
**Next Phase**: Phase 2 (Modern Architectures) starts Week 3
