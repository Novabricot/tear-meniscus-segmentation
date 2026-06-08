# Implementation Notes - Tear Meniscus Segmentation

**Project Start Date**: May 25, 2026  
**Status**: Preparing Phase 1  

## Week 1: Setup & Initial Preparation

### Day 1-2: Environment Setup
- [ ] Python virtual environment created
- [ ] Dependencies installed (see requirements.txt)
- [ ] GPU verified and tested
- [ ] Dataset downloaded (11 GB from Figshare)

### Day 3-4: Data Preparation
- [ ] Dataset extracted to `data/raw/`
  - Color images: 1,693 files
  - IR images: 1,739 files
  - Annotations: masks and pupillary area labels
- [ ] Train/val/test split created (70/15/15)
  - Training: ~2,400 images
  - Validation: ~500 images
  - Testing: ~500 images
- [ ] Data loader implemented and tested

### Day 5: Code Development
- [ ] Augmentation pipeline implemented
  - ElasticTransform, GridDistortion
  - Brightness/Contrast, Gamma
  - Noise augmentation (Gaussian, ISO)
  - MixUp for segmentation
- [ ] Metrics calculation functions
  - F1, Recall, Precision, MIoU
- [ ] Training loop skeleton created
- [ ] TensorBoard logging setup

**Notes**:
- Dataset quality looks good, no obvious corrupted images
- Placido ring reflections present in all color images (important for Phase 3B)
- Aspect ratio 1024×1360 (not square), will need careful resizing in Phase 2

---

## Week 2: Phase 1 Training

### Day 1-2: Model Building & Setup
- [ ] DeepLabV3+ ResNet-101 model loaded
  - Pretrained ImageNet weights
  - 2 output classes (background + tear meniscus)
- [ ] Loss function: Dice Loss (good for binary segmentation)
- [ ] Optimizer: Adam with lr=0.001
- [ ] Scheduler: CosineAnnealingLR with warmup

### Day 3-4: Training Execution
- [ ] Training started with 100 epochs
- [ ] Batch size: 8
- [ ] GPU utilization: ~85% (good)
- [ ] Training time: ~15 hours expected
  - Each epoch: ~9 minutes
  - Validation: ~1 minute per epoch

### Day 5: Evaluation
- [ ] Best model checkpoint saved
- [ ] Test set evaluation
- [ ] Metrics recorded

**Training Observations**:
- [To be filled during training]
- Convergence speed: [TBD]
- Best validation F1 achieved: [TBD]
- Issues encountered: [TBD]

---

## Week 3-4: Phase 2 - SegFormer

### Planned Architecture
- **Model**: SegFormer-B3
- **Input size**: 512×512 (resized from 1024×1360)
- **Batch size**: 4 (with gradient accumulation×2)
- **Learning rate**: 5e-5 (lower for fine-tuning)
- **Scheduler**: PolynomialDecay
- **Expected F1**: 0.93-0.94

### Setup Progress
- [ ] Transformers library installed
- [ ] SegFormer-B3 weights downloaded
- [ ] Input preprocessing pipeline (aspect ratio preservation)
- [ ] Gradient accumulation implementation
- [ ] Training script implemented

---

## Week 5-9: Phase 3 - Advanced Techniques

### Phase 3A: SimCLR Pretraining
- [ ] Contrastive learning setup
- [ ] All 3,432 images (labeled + unlabeled IR)
- [ ] Expected pretraining time: 7-10 days
- [ ] Expected F1 gain: +2%

### Phase 3B: Domain Adversarial Training
- [ ] Domain discriminator implementation
- [ ] Ring masking for synthetic domain data
- [ ] Adversarial loss weighting
- [ ] Expected robustness improvement: 10% → 5% drop on masked images

### Phase 3C: Ensemble Methods
- [ ] Retraining SegFormer-B3 with 3 different random seeds
- [ ] Ensemble averaging implementation
- [ ] Uncertainty estimation
- [ ] Expected F1: 0.93-0.94 (more stable)

---

## Key Decisions Made

### Architecture Progression
1. **Phase 1**: ResNet-101 (upgrade baseline, low risk)
2. **Phase 2**: SegFormer-B3 (transformer, better global context)
3. **Phase 3**: Multiple techniques (pretraining, domain adaptation, ensemble)

### Why SegFormer over alternatives?
- SETR: Too slow (20 FPS vs 80 FPS)
- Mask2Former: Too heavy (requires 16 GB GPU)
- UNetFormer: Less mature, fewer examples
- SegFormer-B3: Sweet spot of accuracy, speed, resources

### Data Augmentation Strategy
- Start moderate (Phase 1), increase gradually (Phase 3)
- Focus on ocular variations (elastic deformation, noise)
- Include domain randomization for ring robustness

---

## Challenges & Solutions

### Challenge 1: Input Size Mismatch
- **Issue**: Images are 1024×1360 (non-square)
- **Solution**: Resize maintaining aspect ratio + pad to 512×512
- **Trade-off**: Small accuracy loss (<1%) for inference speed

### Challenge 2: Placido Ring Dependency
- **Issue**: Model may learn rings as features
- **Solution**: Phase 3B domain adversarial training
- **Timeline**: Add after Phase 2 baseline

### Challenge 3: Limited Dataset Size
- **Issue**: 3,432 images is relatively small for transformers
- **Solution**: Phase 3A self-supervised pretraining on all images
- **Benefit**: Uses unlabeled IR images effectively

---

## Performance Targets (Updated)

| Phase | Model | Target F1 | Status |
|---|---|---|---|
| Baseline | U-Net | 0.9220 | Reference |
| 1 | ResNet-101 | 0.90-0.91 | [TBD] |
| 2 | SegFormer-B3 | 0.93-0.94 | [TBD] |
| 3A | + SimCLR | 0.94-0.95 | [TBD] |
| 3B | + Domain Adv | 0.94-0.95 | [TBD] |
| 3C | + Ensemble | 0.94-0.96 | [TBD] |

---

## Git Commits & Milestones

### Completed
- [x] Repository created: `tear-meniscus-segmentation`
- [x] Project structure initialized
- [x] Documentation & guides prepared
- [x] .gitignore configured

### In Progress
- [ ] Phase 1 training
- [ ] Initial results

### Planned
- [ ] Phase 1 complete & committed (Week 2)
- [ ] Phase 2 start & committed (Week 3)
- [ ] Phase 2 complete & committed (Week 4)
- [ ] Phase 3 phases start (Week 5)
- [ ] Final results & paper submission (Week 9)

---

## Resources Used

### Datasets
- Primary: Figshare Tear Meniscus Segmentation (3,432 images)
- Device: Oculus Keratograph 5M
- Centers: 5 clinical centers in China

### Libraries & Tools
- **Deep Learning**: PyTorch 2.0
- **Segmentation Models**: `segmentation-models-pytorch`
- **Transformers**: Hugging Face Transformers
- **Augmentation**: Albumentations
- **Self-supervised**: Lightly
- **Logging**: TensorBoard, Weights & Biases

### References
- SegFormer Paper: https://arxiv.org/abs/2105.03902
- Original Tear Meniscus Dataset: https://doi.org/10.6084/m9.figshare.28650536.v2

---

## Next Steps

1. **This Week**: Start Phase 1 training
2. **Week 2**: Complete Phase 1, document results
3. **Week 3**: Prepare and start Phase 2
4. **Weeks 5-9**: Complete Phase 3 techniques
5. **Week 10**: Final evaluation, paper preparation

---

**Last Updated**: May 25, 2026  
**Next Review**: May 30, 2026 (Day 5, end of Week 1)  
**Author**: Elora (@Novabricot)
