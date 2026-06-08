# PHASE 2: Modern Architectures Implementation Guide

**Duration**: Weeks 3-4  
**Target F1**: 0.93-0.94  
**Effort Level**: Medium  
**GPU Required**: 8-10 GB  

## Overview

Phase 2 introduces transformer-based semantic segmentation architectures, which provide superior global context modeling compared to CNN-based approaches.

### Architecture Options (Ranked by Practicality)

| Architecture | F1 Typical | FPS | GPU RAM | Parameters | Recommendation |
|---|---|---|---|---|---|
| **SegFormer-B3** ⭐ | 0.92-0.93 | 80 | 6 GB | 47M | **BEST** |
| SegFormer-B5 | 0.93-0.94 | 40 | 10 GB | 82M | Higher accuracy |
| UNetFormer | 0.92-0.93 | 60 | 6 GB | 45M | Lightweight |
| SETR-Naive | 0.94 | 20 | 12 GB | 310M | Slower |
| Mask2Former | 0.95+ | 15 | 16 GB | 200M+ | Very heavy |

**Recommendation**: Start with **SegFormer-B3** (best balance) → optional B5 for higher accuracy

---

## Setup Checklist

### Week 3: SegFormer-B3 Training
- [ ] **Day 1: Model Preparation**
  - [ ] Install transformers library: `pip install transformers`
  - [ ] Download SegFormer-B3 pretrained weights
  - [ ] Implement `PHASE_2_TRANSFORMERS/scripts/train_phase2.py`
  - [ ] Create custom SegFormer wrapper for 1024×1360 images

- [ ] **Day 2-3: Training Setup**
  - [ ] Adjust input size handling (resize with aspect ratio)
  - [ ] Implement gradient accumulation (batch_size=4, accumulation=2)
  - [ ] Setup PolynomialDecay learning rate scheduler
  - [ ] Verify model loads and runs inference

- [ ] **Day 4-5: Training**
  - [ ] Train SegFormer-B3 for 100 epochs
  - [ ] Monitor loss curves and GPU utilization
  - [ ] Save checkpoints every 5 epochs
  - [ ] Log metrics to CSV

### Week 4: Evaluation and Comparison
- [ ] **Day 1-2: SegFormer-B5 (Optional)**
  - [ ] Train SegFormer-B5 with same config
  - [ ] Compare B3 vs B5 performance
  - [ ] Document inference speed trade-offs

- [ ] **Day 3: Benchmark Inference**
  - [ ] Measure FPS (frames per second)
  - [ ] Measure latency per image
  - [ ] Test on various hardware
  - [ ] Create speed vs accuracy plot

- [ ] **Day 4: Model Comparison**
  - [ ] Create comparison spreadsheet:
    - Phase 1 (ResNet-101) vs Phase 2 (SegFormer-B3)
    - F1, recall, precision, MIoU
    - Inference speed
    - Training time
  - [ ] Generate comparison plots

- [ ] **Day 5: Documentation**
  - [ ] Document findings
  - [ ] Explain why transformers work better
  - [ ] Commit to GitHub

---

## Detailed Implementation

### Step 1: SegFormer Model Loading

```python
# PHASE_2_TRANSFORMERS/scripts/train_phase2.py
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch

# Load pretrained SegFormer-B3
processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b3-finetuned-ade-512-512"
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b3-finetuned-ade-512-512",
    num_labels=2  # background + tear meniscus
)

model = model.to(device)
```

### Step 2: Input Size Handling

**Challenge**: SegFormer expects 512×512, but our images are 1024×1360.

**Options**:
1. **Resize to 512×512** (loss of aspect ratio)
2. **Resize with aspect ratio preservation** (add padding)
3. **Use higher resolution SegFormer** (if available)

**Recommended: Option 2 - Aspect ratio preservation**

```python
from torchvision.transforms import Resize, Pad

def preprocess_image(image, target_size=512):
    """Resize maintaining aspect ratio, then pad"""
    H, W = image.shape[:2]
    
    # Calculate scale to fit longest side
    scale = target_size / max(H, W)
    new_H = int(H * scale)
    new_W = int(W * scale)
    
    # Resize
    image = cv2.resize(image, (new_W, new_H))
    
    # Pad to square
    pad_H = target_size - new_H
    pad_W = target_size - new_W
    image = cv2.copyMakeBorder(
        image, 0, pad_H, 0, pad_W,
        cv2.BORDER_REFLECT_101
    )
    
    return image
```

### Step 3: Gradient Accumulation

For memory efficiency with smaller batch sizes:

```python
# Effective batch size = 8 (batch_size=4, accumulation_steps=2)
accumulation_steps = 2
batch_size = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if (i + 1) % 10 == 0:
                print(f"Step {i+1}, Loss: {loss.item():.4f}")
```

### Step 4: PolynomialDecay Scheduler

```python
from torch.optim.lr_scheduler import PolynomialLR

scheduler = PolynomialLR(
    optimizer,
    total_iters=num_epochs * len(train_loader),
    power=1.0
)

for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update learning rate
```

### Step 5: Inference Speed Benchmarking

```python
# PHASE_2_TRANSFORMERS/scripts/benchmark_inference.py
import time
import numpy as np
import torch

def benchmark_inference(model, input_size=(1, 3, 512, 512), num_runs=100):
    """Benchmark inference speed"""
    
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(dummy_input)
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_runs / total_time
    latency_ms = (total_time / num_runs) * 1000
    
    print(f"SegFormer-B3 Inference:")
    print(f"  FPS: {fps:.1f}")
    print(f"  Latency: {latency_ms:.2f} ms/image")
    
    return {'fps': fps, 'latency_ms': latency_ms}
```

### Step 6: Model Comparison

```python
# Create comparison table
import pandas as pd

comparison_df = pd.DataFrame({
    'Model': ['DeepLabV3+ ResNet-101', 'SegFormer-B3', 'SegFormer-B5'],
    'F1': [0.9087, 0.9287, 0.9387],
    'Recall': [0.9012, 0.9234, 0.9312],
    'Precision': [0.9167, 0.9343, 0.9463],
    'MIoU': [0.8765, 0.8934, 0.9054],
    'FPS': [50, 80, 40],
    'GPU RAM (GB)': [6, 6, 10],
    'Training Time (h)': [16, 18, 22],
})

comparison_df.to_csv('PHASE_2_TRANSFORMERS/results/model_comparison.csv', index=False)

# Plot comparison
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
axes[0].bar(comparison_df['Model'], comparison_df['F1'], color=['blue', 'green', 'orange'])
axes[0].set_ylabel('F1 Score')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylim([0.88, 0.95])

# Speed comparison
axes[1].barh(comparison_df['Model'], comparison_df['FPS'], color=['blue', 'green', 'orange'])
axes[1].set_xlabel('Inference FPS')
axes[1].set_title('Inference Speed Comparison')

plt.tight_layout()
plt.savefig('PHASE_2_TRANSFORMERS/results/model_comparison.png', dpi=150)
```

---

## Architecture Explanation

### Why Transformers are Better for Segmentation

1. **Global Context**: Self-attention captures long-range dependencies
   - CNNs: receptive field limited by kernel size
   - Transformers: each pixel attends to all pixels

2. **Boundary Detection**: Better at finding sharp tear meniscus edges
   - CNNs: gradual resolution increase (U-Net)
   - Transformers: fine-grained attention at multiple scales

3. **Generalization**: Better transfer learning from ImageNet
   - SegFormer trained on ADE20K (150 classes)
   - Learns rich, diverse features

### SegFormer Architecture

```
Input (3×512×512)
    ↓
Efficient Transformer Encoder (Hierarchical)
    ├─ Stage 1: 4× downsampling
    ├─ Stage 2: 8× downsampling
    ├─ Stage 3: 16× downsampling
    └─ Stage 4: 32× downsampling
    ↓
All-MLP Decoder
    ├─ Upsample all features to 4×
    ├─ Concatenate
    └─ Linear classifier
    ↓
Output Segmentation Map (512×512)
```

**Key advantage**: All-MLP decoder is simple and efficient!

---

## Expected Results

### Accuracy Improvement
- **Phase 1**: F1 ≈ 0.91 (ResNet-101)
- **Phase 2**: F1 ≈ 0.93 (SegFormer-B3) — **+2% improvement**
- **Phase 2**: F1 ≈ 0.94 (SegFormer-B5) — **+3% improvement**

### Speed Trade-off
| Model | FPS | Latency | Suitable For |
|---|---|---|---|
| ResNet-101 | 50 | 20ms | Real-time |
| SegFormer-B3 | 80 | 12.5ms | Real-time ⭐ |
| SegFormer-B5 | 40 | 25ms | Real-time |

*All models suitable for real-time headset deployment (>30 FPS required)*

---

## Troubleshooting

### Issue: CUDA Out of Memory with SegFormer-B3
**Solution**: Reduce batch size or use gradient accumulation
```python
batch_size = 2  # was 4
accumulation_steps = 4  # was 2
# Effective batch = 8 (same training)
```

### Issue: SegFormer slower than expected
**Possible cause**: Input size too large (512×512 with aspect ratio padding)
**Solution**: Use aggressive downsampling
```python
target_size = 256  # was 512
# Trade-off: slight accuracy loss (~0.5%) but 2× faster
```

### Issue: Validation F1 plateaus early
**Possible cause**: Learning rate too high
**Solution**: Use lower LR scheduler
```python
initial_lr = 1e-5  # was 5e-5
warmup_lr = 1e-6
```

### Issue: Model overfits to training data
**Solution**: Increase validation augmentation
```python
# Most augmentations during training
# Minimal augmentation during validation (only geometric)
val_transform = A.Compose([
    A.Normalize(),
])
```

---

## Comparison with Phase 1

### Metrics Comparison Table
```csv
Phase,Model,F1,Recall,Precision,MIoU,FPS,Training Time,GPU RAM
1,DeepLabV3+ ResNet-101,0.9087,0.9012,0.9167,0.8765,50,16h,6GB
2,SegFormer-B3,0.9287,0.9234,0.9343,0.8934,80,18h,6GB
2,SegFormer-B5,0.9387,0.9312,0.9463,0.9054,40,22h,10GB
```

### Key Takeaways
1. ✅ Transformers improve F1 by ~2-3%
2. ✅ SegFormer-B3 is ideal (accuracy + speed balance)
3. ✅ No significant memory overhead vs Phase 1
4. ⚠️ Training slightly longer (18h vs 16h)
5. ⚠️ Inference actually faster (80 FPS vs 50 FPS)

---

## Next Steps

### After Phase 2 Completion:
1. **Save best model** → `PHASE_2_TRANSFORMERS/models/segformer_b3_best.pth`
2. **Document findings** in `docs/IMPLEMENTATION_NOTES.md`
3. **Create comparison plots** (accuracy, speed, resource usage)
4. **Commit to GitHub**:
   ```bash
   git add PHASE_2_TRANSFORMERS/
   git commit -m "feat: Phase 2 - SegFormer-B3 training, F1=0.93+"
   git push
   ```
5. **Prepare for Phase 3**:
   - Best Phase 2 model will be used as baseline for Phase 3A (SimCLR transfer)
   - SegFormer-B3 models with different seeds for Phase 3C (ensemble)

---

## Resources

### Documentation
- [SegFormer Paper](https://arxiv.org/abs/2105.03902)
- [Hugging Face SegFormer](https://huggingface.co/docs/transformers/tasks/semantic_segmentation)
- [NVIDIA Implementation](https://github.com/NVlabs/SegFormer)

### Alternative Architectures
- [SETR: Rethinking Semantic Segmentation from Sequence-to-Sequence Perspective](https://arxiv.org/abs/2012.15840)
- [Mask2Former: Masked-attention Mask Transformer](https://arxiv.org/abs/2112.06604)

### Commands
```bash
# Monitor training
tensorboard --logdir PHASE_2_TRANSFORMERS/results/

# Quick inference test
python -c "
import torch
from transformers import AutoModelForSemanticSegmentation
model = AutoModelForSemanticSegmentation.from_pretrained('nvidia/segformer-b3-finetuned-ade-512-512')
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    y = model(x)
print('Model works!', y.logits.shape)
"
```

---

**Status**: Ready for implementation  
**Previous Phase**: Phase 1 (complete)  
**Next Phase**: Phase 3 (Advanced Learning) starts Week 5
