# Tear Meniscus Segmentation: Improvement Strategy

**Project**: Research-phase improvement of tear meniscus semantic segmentation model  
**Date**: May 25, 2026  
**Target**: Improve from baseline U-Net F1=0.9220 to 0.93-0.96 F1  
**Scope**: Research optimization (no headset deployment yet)

---

## 1. BASELINE PERFORMANCE & GOALS

### Current Results
| Model | F1 Score | Recall | Precision | MIoU | Source |
|---|---|---|---|---|---|
| U-Net | 0.9220 | 0.9127 | 0.9367 | 0.9279 | Published paper |
| ResUNet | 0.9025 | 0.8840 | 0.9286 | 0.9109 | Published paper |
| DeepLabV3+ | 0.8705 | 0.8533 | 0.8957 | 0.8841 | Published paper |

### Improvement Targets
| Target Level | F1 Score Range | Classification | Effort |
|---|---|---|---|
| Conservative (Phase 1+2) | 0.93-0.94 | Publishable improvement | 3-4 weeks |
| Aggressive (All phases) | 0.94-0.96 | Strong research contribution | 7-9 weeks |
| "Satisfying" threshold | 0.93+ | Clinically meaningful improvement | 3+ weeks |

**Definition of Success**: Achieve ≥0.93 F1 with measurable improvement over baseline.

---

## 2. DATASET INFORMATION

### Current Dataset
- **Name**: Pixel-Level Tear Meniscus Segmentation Dataset with Multimodal Imaging
- **Source**: Figshare (https://doi.org/10.6084/m9.figshare.28650536.v2)
- **Size**:
  - 1,693 color images
  - 1,739 infrared (IR) images
  - Total: 3,432 images
  - Format: PNG, 1024 × 1360 pixels
  
- **Storage requirement for raw dataset**:
  ```
  Color images: 1,693 × (1024 × 1360 × 3 bytes RGB) = 1,693 × ~4.2 MB ≈ 7.1 GB
  IR images: 1,739 × (1024 × 1360 × 1 byte grayscale) = 1,739 × ~1.4 MB ≈ 2.4 GB
  Masks/annotations: ~1.5 GB
  
  Total (uncompressed): ~11 GB
  ```

- **Labels**: 
  - Lower tear meniscus segmentation masks
  - Central pupillary area masks
  - Both color and IR images have corresponding masks

- **Acquisition details**:
  - Device: Oculus Keratograph 5M
  - 5 clinical centers in China
  - Contains Placido ring reflections (important for domain adaptation)

---

## 3. IMPROVEMENT STRATEGY: 3 PHASES

### PHASE 1: QUICK WINS (Weeks 1-2)
**Goal**: Achieve 0.90-0.91 F1 with minimal computational overhead  
**Estimated improvement**: +2-3% over baseline

#### A. Architecture Upgrade: DeepLabV3+ ResNet-101
- **Current**: DeepLabV3+ with ResNet-50 (0.8705 F1)
- **Target**: DeepLabV3+ with ResNet-101
- **Why**: Larger backbone captures fine features better, especially for boundary detection

**Implementation**:
```python
# PyTorch code template
import segmentation_models_pytorch as smp

# Current
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# Upgraded
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",  # Larger backbone
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
```

**Storage requirements**:
- ResNet-50 weights: ~98 MB
- ResNet-101 weights: ~170 MB
- Model on GPU: ~800 MB (inference only)
- Training memory: 6-8 GB GPU RAM required

**Expected improvement**: +1-2% F1

---

#### B. Enhanced Data Augmentation
Implement multi-stage augmentation pipeline with focus on ocular variations.

**Augmentation techniques** (add to training pipeline):
1. **Color/Intensity variations**:
   - RandomBrightnessContrast (p=0.3): IR images have different intensity distributions
   - RandomGamma (p=0.2): Simulate different exposure levels
   - CLAHE (Contrast Limited Adaptive Histogram Equalization): Normalize tear meniscus contrast

2. **Geometric transformations**:
   - ElasticTransform (p=0.3, sigma=5): Simulate natural eye deformation
   - GridDistortion (p=0.2): Non-uniform spatial variations

3. **Domain randomization**:
   - GaussianNoise (p=0.2, var_limit=(10, 50)): Sensor noise variation
   - ISONoise (p=0.1): Camera ISO variation

**Implementation** (using albumentations library):
```python
import albumentations as A

train_transform = A.Compose([
    # Color/Intensity
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    A.CLAHE(p=0.2),
    
    # Geometric
    A.ElasticTransform(alpha=1, sigma=5, alpha_affine=5, p=0.3),
    A.GridDistortion(p=0.2),
    
    # Domain
    A.GaussNoise(p=0.2),
    A.ISONoise(p=0.1),
    
    # Standard
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.2),
], keypoint_params=A.KeypointParams(format='xy'))
```

**Storage requirements**: None (computed on-the-fly during training)

**Expected improvement**: +1-2% F1

---

#### C. Progressive Augmentation Schedule
Gradually increase augmentation intensity during training epochs.

**Implementation**:
```python
def get_augmentation_intensity(epoch, total_epochs):
    """Increase augmentation as training progresses"""
    intensity = epoch / total_epochs
    return {
        'brightness_limit': 0.1 + (intensity * 0.15),
        'contrast_limit': 0.1 + (intensity * 0.15),
        'noise_var': 10 + (intensity * 30),
        'elastic_alpha': 1 + (intensity * 3),
    }

# Use in training loop
for epoch in range(total_epochs):
    aug_params = get_augmentation_intensity(epoch, total_epochs)
    train_dataloader.dataset.transform = build_augmentation(aug_params)
    train_one_epoch(model, train_dataloader, optimizer, loss_fn)
```

**Storage requirements**: None

**Expected improvement**: +0.5-1.5% F1

---

### PHASE 1 SUMMARY
**Total expected improvement**: +2-4% F1 (target: 0.90-0.94 F1)  
**Implementation time**: 1-2 weeks  
**Storage needed**: ~1-2 GB (for models + training checkpoints)  
**GPU memory needed**: 6-8 GB  
**CPU memory needed**: 16 GB (for batch loading)

**Deliverables**:
- Upgraded DeepLabV3+ ResNet-101 model
- Enhanced augmentation configuration file
- Training logs with loss curves

---

### PHASE 2: MODERN ARCHITECTURE (Weeks 3-4)
**Goal**: Achieve 0.93-0.94 F1 with transformer-based architecture  
**Estimated improvement**: +2-4% over Phase 1

#### Option A: SegFormer (RECOMMENDED)
**Why SegFormer**:
- Efficient transformer backbone (MIT-B0 to B5)
- 80-100 FPS inference on modern GPUs
- Superior global context compared to CNNs
- Best balance of accuracy vs. computational cost

**Architecture comparison**:
| Architecture | F1 (typical) | Inference FPS | GPU RAM | Parameters |
|---|---|---|---|---|
| DeepLabV3+ ResNet-50 | 0.87 | 50 | 4 GB | 40M |
| DeepLabV3+ ResNet-101 | 0.89 | 35 | 6 GB | 60M |
| **SegFormer B3** | 0.92-0.93 | 80 | 6 GB | 47M |
| **SegFormer B5** | 0.93-0.94 | 40 | 10 GB | 82M |
| SETR-Naive | 0.94 | 20 | 12 GB | 310M |

**Recommended**: SegFormer-B3 or B5 (good balance of accuracy and inference speed)

**Implementation**:
```python
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# Load pretrained SegFormer
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512", 
                                                         num_labels=2)  # background + tear meniscus

# Fine-tune on tear meniscus dataset
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = processor(batch['image'], segmentation_maps=batch['mask'], return_tensors="pt")
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**Storage requirements**:
- SegFormer-B3 pretrained weights: ~180 MB
- SegFormer-B5 pretrained weights: ~360 MB
- Training checkpoints (5-10 per phase): 0.9-3.6 GB
- **Total**: 1-4 GB depending on model choice

**Expected improvement**: +2-4% F1 (cumulative: 0.92-0.94 F1)

---

#### Option B: UNetFormer (ALTERNATIVE)
**Why UNetFormer**:
- Hybrid CNN-Transformer architecture
- More efficient than pure transformers
- Good for medical imaging (designed for segmentation)
- Smaller than SegFormer B5

**Specs**:
- Inference: 60-80 FPS
- GPU RAM: 6-8 GB
- Parameters: 45M
- F1 expected: 0.92-0.93

**Implementation**:
```python
from timm import create_model

model = create_model('unetformer', pretrained=True, num_classes=2)
# Fine-tune similarly to SegFormer
```

**Storage requirements**: 0.8-2 GB (smaller than SegFormer)

**Expected improvement**: +1.5-3% F1

---

#### Option C: Mask2Former (SOTA)
**Why Mask2Former**:
- Latest state-of-the-art (2023)
- Unified masked attention for all segmentation tasks
- Excellent boundary detection
- Higher accuracy but computationally heavier

**Specs**:
- Inference: 20-30 FPS
- GPU RAM: 12-16 GB
- Parameters: 100-200M depending on backbone
- F1 expected: 0.94+

**When to use**: If computational resources not a concern and maximum accuracy desired

**Storage requirements**: 2-5 GB

**Expected improvement**: +2-4% F1

---

### PHASE 2 SUMMARY
**Recommended model**: SegFormer-B3 (best balance)  
**Total expected improvement**: +2-4% F1 (cumulative target: 0.93-0.94 F1)  
**Implementation time**: 2-3 weeks  
**Storage needed**: 1-4 GB (models + checkpoints)  
**GPU memory needed**: 8-10 GB  
**CPU memory needed**: 16-32 GB

**Deliverables**:
- Fine-tuned SegFormer model
- Comparative performance report (DeepLabV3+ vs SegFormer)
- Inference speed benchmarks

---

### PHASE 3: ADVANCED LEARNING (Weeks 5-9)
**Goal**: Achieve 0.94-0.96 F1 with advanced training techniques  
**Estimated improvement**: +2-4% over Phase 2

#### A. Self-Supervised Pretraining on IR Images
**Motivation**: You have 1,739 unlabeled IR images. Use them for representation learning.

**Approach**: SimCLR (Simple Framework for Contrastive Learning)

**How it works**:
1. Create two random augmentations of each image
2. Pass through encoder → get embeddings
3. Maximize similarity of same-image pairs
4. Minimize similarity of different-image pairs
5. Transfer learned encoder to segmentation task

**Implementation**:
```python
from lightly.models import SimCLR
from lightly.loss import NTXentLoss

# Step 1: Contrastive pretraining on ALL 3,432 images
backbone = torchvision.models.resnet50()
model = SimCLR(backbone, hidden_dim=2048, out_dim=128)

contrastive_loss = NTXentLoss()
for epoch in range(100):  # Pretrain for 100 epochs
    for batch in all_images_dataloader:  # Use all images, no labels needed
        # Create augmented pair
        x0, x1 = batch
        z0 = model(x0)
        z1 = model(x1)
        loss = contrastive_loss(z0, z1)
        loss.backward()

# Step 2: Transfer to segmentation task
# Use pretrained ResNet-50 encoder in SegFormer
model_segmentation = SegFormer(encoder='resnet50-pretrained')
# Fine-tune on labeled tear meniscus data only
```

**Storage requirements**:
- Pretraining checkpoint: ~200 MB
- Final segmentation model: 180 MB
- **Total**: 0.5-1 GB

**Expected improvement**: +2-4% F1 (uses 100% of available imagery)

**Timeline**: 1-2 weeks (pretraining: ~7-10 days on modern GPU)

---

#### B. Domain Adversarial Training
**Motivation**: Make model robust to Placido ring variations (headset-safe, even though headset not immediate target)

**Problem**: Current model may rely on ring patterns as cues. Need features invariant to this domain shift.

**Approach**:
1. Train segmentation model (as before)
2. Add domain discriminator head
3. Discriminator predicts: "Has Placido rings?" vs "No rings"
4. Adversarial loss forces segmentation features to ignore rings

**Implementation**:
```python
class DomainAdversarialSegmentation(nn.Module):
    def __init__(self, segmentation_backbone, hidden_dim=256):
        super().__init__()
        self.encoder = segmentation_backbone.encoder
        self.decoder = segmentation_backbone.decoder
        
        # Segmentation task
        self.segmentation_head = nn.Conv2d(hidden_dim, 2, 1)
        
        # Domain discriminator (adversarial)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # binary: has rings or not
        )
    
    def forward(self, x, return_domain=False):
        features = self.encoder(x)
        seg_output = self.decoder(features)
        seg_output = self.segmentation_head(seg_output)
        
        if return_domain:
            domain_output = self.domain_head(features.mean(dim=[2, 3]))
            return seg_output, domain_output
        return seg_output

# Training with adversarial loss
lambda_adv = 0.1  # Weight of adversarial loss
for epoch in range(num_epochs):
    for batch in dataloader:
        # Segmentation loss (as normal)
        seg_pred, domain_pred = model(batch['image'], return_domain=True)
        seg_loss = dice_loss(seg_pred, batch['mask'])
        
        # Adversarial loss: make features domain-invariant
        domain_loss = ce_loss(domain_pred, batch['domain_label'])  # 0=no rings, 1=rings
        
        total_loss = seg_loss - lambda_adv * domain_loss  # Negative: fool discriminator
        total_loss.backward()
        optimizer.step()
```

**Data preparation for domain adversarial**:
- Create masked versions of color images (remove rings artificially)
- Label original as domain=1 (has rings), masked as domain=0 (no rings)
- This trains robustness without additional data collection

**Storage requirements**: 
- No additional storage (uses existing images)
- Model size: same as base SegFormer (~180-360 MB)

**Expected improvement**: +0.5-1.5% F1 (primarily robustness, not raw accuracy)

**Timeline**: 1-2 weeks

---

#### C. Ensemble Methods
**Simple but effective**: Combine multiple trained models for robustness

**Approach**:
1. Train SegFormer-B3 with different random seeds (3-5 models)
2. Train SegFormer-B5 (1 model)
3. Average predictions at inference

**Implementation**:
```python
class EnsembleSegmentation:
    def __init__(self, model_paths):
        self.models = [torch.load(p) for p in model_paths]
        for m in self.models:
            m.eval()
    
    def predict(self, image):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(image)
                predictions.append(torch.softmax(pred, dim=1))
        
        # Average ensemble
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

ensemble = EnsembleSegmentation([
    'checkpoint_b3_seed0.pth',
    'checkpoint_b3_seed1.pth',
    'checkpoint_b3_seed2.pth',
    'checkpoint_b5.pth',
])
```

**Storage requirements**:
- 4-5 model checkpoints: 0.7-1.8 GB

**Expected improvement**: +0.5-1% F1 (stability, not raw accuracy)

**Timeline**: 1 week (reuse trained models from Phase 2)

---

### PHASE 3 SUMMARY
**Total expected improvement**: +2-4% F1 (cumulative target: 0.94-0.96 F1)  
**Implementation time**: 3-4 weeks (can parallelize some tasks)  
**Storage needed**: 1-3 GB (pretraining + models + checkpoints)  
**GPU memory needed**: 8-16 GB (pretraining needs more)  
**CPU memory needed**: 32 GB recommended

**Deliverables**:
- Self-supervised pretraining pipeline (transferable to new tasks)
- Domain-adversarial segmentation model
- Ensemble prediction system
- Comprehensive comparison paper

---

## 4. DETAILED STORAGE & RESOURCE REQUIREMENTS

### Storage Breakdown

```
Initial Setup:
├── Raw Dataset (downloaded from Figshare)
│   ├── Color images (1,693 × ~4.2 MB)           : 7.1 GB
│   ├── IR images (1,739 × ~1.4 MB)               : 2.4 GB
│   ├── Masks & annotations                       : 1.5 GB
│   └── Subtotal                                  : ~11 GB

Phase 1 (Weeks 1-2):
├── DeepLabV3+ ResNet-101 weights                 : 0.17 GB
├── Training checkpoints (10 files)               : 1.0 GB
├── Training logs, results                        : 0.1 GB
└── Subtotal                                      : ~1.3 GB

Phase 2 (Weeks 3-4):
├── SegFormer-B3 weights                          : 0.18 GB
├── SegFormer-B5 weights                          : 0.36 GB
├── Training checkpoints (20 files)               : 3.6 GB
├── Comparative benchmarks                        : 0.2 GB
└── Subtotal                                      : ~4.3 GB

Phase 3 (Weeks 5-9):
├── SimCLR pretraining checkpoint                 : 0.2 GB
├── Pretrained encoder transfer                   : 0.2 GB
├── Domain adversarial models (5 versions)        : 0.9 GB
├── Ensemble models (4-5 models)                  : 0.7-1.8 GB
├── Training logs & analysis                      : 0.3 GB
└── Subtotal                                      : ~2.9 GB

Final Outputs:
├── Best performing model                         : 0.36 GB
├── Inference scripts & configs                   : 0.05 GB
├── Results & analysis tables                     : 0.1 GB
├── Paper/documentation                           : 0.05 GB
└── Subtotal                                      : ~0.56 GB

────────────────────────────────────────────────────
TOTAL STORAGE NEEDED: ~20 GB (for all phases + dataset)
```

### Recommended Disk Setup
```
Primary working drive (SSD recommended):
├── Dataset                                       : 11 GB
├── Working models & checkpoints                  : 8 GB
└── Results & outputs                             : 1 GB
   
SUBTOTAL: ~20 GB minimum
RECOMMENDED: 30-40 GB (extra buffer for temporary files)

Secondary backup drive (HDD acceptable):
├── Backup of original dataset                    : 11 GB
├── Final best models                             : 0.5 GB
└── Complete training logs                        : 0.5 GB
```

### GPU Memory Requirements
```
Phase 1 (DeepLabV3+ ResNet-101):
├── Model weights on GPU                          : 0.6 GB
├── Batch size 8 (1024×1360 images)               : 4.8 GB
├── Optimizer states                              : 1.2 GB
└── TOTAL GPU RAM NEEDED: 6-8 GB

Phase 2 (SegFormer-B3):
├── Model weights on GPU                          : 0.5 GB
├── Batch size 4-6                                : 6-8 GB
├── Optimizer states                              : 1.0 GB
└── TOTAL GPU RAM NEEDED: 8-10 GB

Phase 3 (Self-supervised pretraining):
├── Model weights on GPU                          : 0.8 GB
├── Batch size 32-64 (contrastive learning)       : 8-12 GB
├── Optimizer states                              : 1.5 GB
└── TOTAL GPU RAM NEEDED: 10-14 GB

RECOMMENDED GPU: RTX 3080 (10 GB) or RTX 4090 (24 GB)
ACCEPTABLE: RTX 3070 (8 GB) with gradient accumulation
```

### CPU Memory Requirements
```
Data loading & augmentation:
├── Batch loading (8-32 images)                   : 2-4 GB
├── Augmentation buffers                          : 1-2 GB
├── Numpy/PyTorch tensors                         : 2-3 GB
└── TOTAL CPU RAM NEEDED: 16-32 GB

RECOMMENDED: 32 GB RAM minimum for Phase 3
ACCEPTABLE: 16 GB RAM with batch size reduction
```

### Compute Time Estimates

```
Phase 1 (DeepLabV3+ ResNet-101):
├── Training (100 epochs, batch 8)                : 12-16 hours
├── Validation & testing                          : 2 hours
└── Total                                         : ~1-2 days (GPU utilization: 80-90%)

Phase 2 (SegFormer-B3):
├── Training (100 epochs, batch 4)                : 20-24 hours
├── Hyperparameter tuning (if needed)             : 8-12 hours
├── Comparative evaluation                        : 3 hours
└── Total                                         : ~1.5-2 weeks (GPU utilization: 70-85%)

Phase 3a (SimCLR Pretraining):
├── Pretraining on 3,432 images (100 epochs)     : 7-10 days
├── Feature extraction & analysis                 : 4 hours
└── Total                                         : ~1.5 weeks (GPU utilization: 85-95%)

Phase 3b (Domain Adversarial Training):
├── Creating ring masks                           : 2 hours
├── Training with adversarial loss                : 16-20 hours
├── Evaluation on masked variants                 : 3 hours
└── Total                                         : 2-3 days

TOTAL PROJECT TIME: 6-9 weeks (with weekly GPU availability)
```

---

## 5. IMPLEMENTATION CHECKLIST

### Pre-Implementation Setup
- [ ] Download dataset from Figshare (~11 GB)
- [ ] Verify available disk space (30-40 GB minimum)
- [ ] Check GPU availability and memory (8-14 GB depending on phase)
- [ ] Install required packages:
  ```bash
  pip install torch torchvision torchaudio
  pip install segmentation-models-pytorch
  pip install albumentations
  pip install timm transformers
  pip install lightly  # For SimCLR pretraining
  pip install tensorboard wandb  # For logging
  pip install scikit-learn opencv-python
  ```

### Phase 1 Checklist (Weeks 1-2)
- [ ] Set up training pipeline with PyTorch Lightning or Hugging Face Trainer
- [ ] Implement enhanced augmentation pipeline (albumentations)
- [ ] Train DeepLabV3+ ResNet-101 baseline
- [ ] Monitor loss curves on TensorBoard
- [ ] Validate on test set, record F1/IoU metrics
- [ ] Save best checkpoint
- [ ] Document results (spreadsheet or markdown)

### Phase 2 Checklist (Weeks 3-4)
- [ ] Download SegFormer-B3 pretrained weights
- [ ] Implement SegFormer fine-tuning pipeline
- [ ] Train SegFormer-B3 with same augmentation as Phase 1
- [ ] Train SegFormer-B5 (if GPU allows)
- [ ] Compare inference speed (FPS) across models
- [ ] Create performance comparison table
- [ ] Identify best-performing model
- [ ] Save checkpoint and inference script

### Phase 3a Checklist (Self-Supervised Learning)
- [ ] Implement SimCLR encoder with augmentation
- [ ] Pretrain on all 3,432 images (no labels needed)
- [ ] Extract and visualize learned representations
- [ ] Transfer encoder to SegFormer
- [ ] Fine-tune segmentation head on labeled data
- [ ] Compare with Phase 2 baseline

### Phase 3b Checklist (Domain Adversarial)
- [ ] Create synthetic ring-masked image variants (script needed)
- [ ] Implement domain discriminator head
- [ ] Train with adversarial loss on full dataset
- [ ] Test robustness by evaluating on ring-masked test set
- [ ] Document domain shift reduction metrics

### Phase 3c Checklist (Ensemble)
- [ ] Retrain SegFormer-B3 with 3-5 different seeds
- [ ] Implement ensemble averaging at inference
- [ ] Benchmark ensemble F1 vs individual models
- [ ] Measure inference latency increase (if any)

### Final Deliverables
- [ ] Best-performing model checkpoint
- [ ] Inference script (single image + batch processing)
- [ ] Training logs and loss curves
- [ ] Performance comparison table (all models/phases)
- [ ] Speed benchmarks (FPS, inference latency)
- [ ] Research paper / technical report
- [ ] Code repository with reproducible training configs

---

## 6. FOLDER STRUCTURE FOR RESEARCH PROJECT

Suggested organization for your "Actual research" folder:

```
Actual research/
├── TEAR_MENISCUS_IMPROVEMENT_STRATEGY.md (THIS FILE)
│
├── dataset/
│   ├── raw/                          # Downloaded from Figshare
│   │   ├── color_images/
│   │   ├── ir_images/
│   │   └── annotations/
│   │
│   └── processed/                    # After splitting/preprocessing
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── phase_1/
│   │   ├── deeplabv3plus_resnet101.pth
│   │   └── training_logs_phase1.txt
│   │
│   ├── phase_2/
│   │   ├── segformer_b3_best.pth
│   │   ├── segformer_b5_best.pth
│   │   └── training_logs_phase2.txt
│   │
│   └── phase_3/
│       ├── simclr_pretrained_encoder.pth
│       ├── domain_adversarial_v1.pth
│       ├── ensemble_models/
│       │   ├── model_seed0.pth
│       │   ├── model_seed1.pth
│       │   └── model_seed2.pth
│       └── training_logs_phase3.txt
│
├── scripts/
│   ├── train.py                      # Main training script
│   ├── eval.py                       # Evaluation & metrics
│   ├── inference.py                  # Single image prediction
│   ├── augmentation.py               # Augmentation pipelines
│   ├── pretraining_simclr.py         # Self-supervised learning
│   ├── domain_adversarial.py         # Domain adaptation
│   └── ensemble.py                   # Ensemble inference
│
├── configs/
│   ├── phase1_config.yaml
│   ├── phase2_config.yaml
│   ├── phase3_config.yaml
│   └── augmentation_config.yaml
│
├── results/
│   ├── phase_1_results.csv
│   ├── phase_2_comparison.csv
│   ├── phase_3_ablation.csv
│   │
│   ├── figures/
│   │   ├── loss_curves_phase1.png
│   │   ├── loss_curves_phase2.png
│   │   ├── model_comparison.png
│   │   └── segmentation_examples/
│   │
│   └── metrics/
│       ├── f1_scores.txt
│       ├── inference_speed.txt
│       └── ablation_study.txt
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_augmentation_preview.ipynb # Visualize augmentations
│   ├── 03_results_analysis.ipynb     # Performance analysis
│   └── 04_model_comparison.ipynb     # Compare models
│
└── documentation/
    ├── IMPLEMENTATION_NOTES.md        # Day-to-day progress
    ├── RESEARCH_FINDINGS.md           # Key discoveries
    ├── PERFORMANCE_REPORT.md          # Final results & conclusions
    └── README.md                      # Quick start guide
```

---

## 7. QUICK START COMMANDS

Once setup is complete:

```bash
# Phase 1: Quick wins
python scripts/train.py --config configs/phase1_config.yaml --phase 1

# Phase 2: SegFormer
python scripts/train.py --config configs/phase2_config.yaml --phase 2

# Phase 3a: Self-supervised pretraining
python scripts/pretraining_simclr.py --config configs/phase3_config.yaml

# Phase 3b: Domain adversarial
python scripts/domain_adversarial.py --config configs/phase3_config.yaml

# Inference on single image
python scripts/inference.py --model models/phase_2/segformer_b3_best.pth --image path/to/image.png

# Evaluation & metrics
python scripts/eval.py --model_path models/phase_2/ --test_data dataset/processed/test/
```

---

## 8. RISK FACTORS & MITIGATION

| Risk | Impact | Mitigation |
|---|---|---|
| Out of storage space | Phase halts | Monitor disk usage weekly; delete old checkpoints |
| OOM (Out of Memory) error | Training crashes | Reduce batch size; use gradient accumulation |
| Overfitting on small validation set | Poor generalization | Increase augmentation; use early stopping |
| Placido ring dependency (domain shift) | Poor headset transfer | Phase 3b domain adversarial training required |
| GPU utilization drops | Slow training | Check for data loading bottleneck; reduce number of workers |
| No improvement beyond Phase 1 | Plateauing returns | Try Phase 2; consider ensemble methods |

---

## 9. NEXT IMMEDIATE STEPS

1. **Check your current setup**:
   - Available disk space: `df -h` (Linux/Mac) or `Get-Volume` (Windows)
   - GPU availability: `nvidia-smi` or `torch.cuda.is_available()`
   - RAM available: System information

2. **Download dataset**:
   - Visit Figshare link provided
   - Download all 3,432 images (~11 GB)
   - Verify integrity if possible

3. **Set up development environment**:
   - Create Python virtual environment
   - Install dependencies from section 5
   - Create folder structure from section 6

4. **Start Phase 1** (Week 1):
   - Implement augmentation pipeline
   - Train DeepLabV3+ ResNet-101
   - Establish baseline metrics

5. **Track progress**:
   - Keep spreadsheet of F1/IoU scores
   - Save loss curves and example predictions
   - Document any issues encountered

---

## 10. CONTACT POINTS & RESOURCES

### Key Papers/References
- SegFormer: https://arxiv.org/abs/2105.03902
- SETR: https://arxiv.org/abs/2012.15840
- Mask2Former: https://arxiv.org/abs/2112.06604
- SimCLR: https://arxiv.org/abs/2002.05709
- Tear Meniscus Dataset: https://doi.org/10.6084/m9.figshare.28650536.v2

### Software Libraries
- Segmentation Models PyTorch: `segmentation-models-pytorch` (all SOTA architectures)
- Albumentations: Image augmentation library
- Hugging Face Transformers: SegFormer, SETR implementations
- Lightly: Self-supervised learning toolkit (SimCLR, BYOL, etc.)

### Monitoring & Visualization
- TensorBoard: Real-time loss/metric monitoring
- Weights & Biases (wandb): Experiment tracking & comparison
- OpenCV: Image processing & visualization

---

**Status**: Ready for implementation  
**Last Updated**: May 25, 2026  
**Owner**: [Your Name]  
**Review Date**: [Set after 2 weeks for Phase 1 results]
