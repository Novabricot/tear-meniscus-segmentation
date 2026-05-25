# Complete GitHub Setup & Push Guide for tear-meniscus-segmentation

**Status**: Project folder created, ready for git & GitHub setup  
**Location**: `c:\Users\Elora\OneDrive - ensicaen.fr\Documents\NAIST\Nouveau dossier\create_research_survey\tear-meniscus-segmentation`  
**Next Step**: Install Git and push to GitHub  

---

## Step 1: Install Git

### Option A: Automatic Installation (Recommended - Windows)

#### Using Winget (Windows 11):
```powershell
winget install Git.Git
```

#### Using Chocolatey:
```powershell
choco install git
```

#### Manual Download:
1. Visit: https://git-scm.com/download/win
2. Download latest version
3. Run installer
4. Use default options (or customize as needed)
5. **Important**: Choose "Add Git to PATH"

### Verify Installation:
```powershell
git --version
# Should output: git version X.XX.X.windows.X
```

If not recognized after installation:
- Restart PowerShell/Command Prompt
- Or add to PATH manually if needed

---

## Step 2: Install GitHub CLI (Easiest for Push)

### Windows Winget:
```powershell
winget install GitHub.cli
```

### Or Chocolatey:
```powershell
choco install gh
```

### Verify:
```powershell
gh --version
# Should output: gh version X.XX.X (YYYY-MM-DD)
```

---

## Step 3: Authenticate with GitHub

### Using GitHub CLI (One Command):
```powershell
gh auth login
```

**Follow prompts:**
1. Select: `GitHub.com`
2. Select: `HTTPS` (easiest)
3. Select: `Yes` when asked to authenticate with browser
4. Browser will open → Authorize `GitHub CLI`
5. Authorize successful!

**Test authentication:**
```powershell
gh auth status
# Should show: Logged in to github.com as Novabricot
```

---

## Step 4: Initialize and Push Project

### After authentication, run these commands:

```powershell
# Navigate to project directory
cd "c:\Users\Elora\OneDrive - ensicaen.fr\Documents\NAIST\Nouveau dossier\create_research_survey\tear-meniscus-segmentation"

# Initialize git repo
git init

# Configure git (if not already globally configured)
git config user.name "Your Name"
git config user.email "your-email@example.com"
# Example:
# git config user.name "Elora"
# git config user.email "elora@naist.ac.jp"

# Add all files (will respect .gitignore)
git add .

# Create initial commit
git commit -m "init: Complete project structure for tear meniscus segmentation

- 3-phase training plan with detailed implementation guides
- Phase 1: DeepLabV3+ ResNet-101 with advanced augmentation
- Phase 2: SegFormer-B3 transformer-based architecture
- Phase 3: SimCLR pretraining, domain adversarial, ensemble methods
- Full configuration files for each phase
- Training scripts templates and utility functions
- Comprehensive documentation and phase guides
- Production-ready .gitignore and CI/CD setup"

# Create repository on GitHub and push
gh repo create tear-meniscus-segmentation \
  --public \
  --source=. \
  --push \
  --description="Improving tear meniscus segmentation for dry eye detection - 3-phase research project with modern architectures and advanced training techniques"
```

**That's it!** Repository created and pushed! 🎉

---

## Verify Success

Visit your GitHub repository:
```
https://github.com/Novabricot/tear-meniscus-segmentation
```

You should see:
- ✅ All project files and folders
- ✅ README.md displayed on main page
- ✅ Complete directory structure
- ✅ GITHUB_SETUP.md and guides included
- ✅ License file visible
- ✅ Green checkmark ✓ on initial commit

---

## Quick Reference: Daily Git Workflow

After setup, your daily workflow for the next 9 weeks:

```powershell
# Before starting work
git pull origin main

# Make changes to code/configs
# ... edit files ...

# Check what changed
git status

# Stage changes (NOT data or models!)
git add scripts/ configs/ PHASE_1_QUICKWINS/ docs/

# Commit with clear message
git commit -m "feat: Phase 1 training improvements

- Implemented enhanced augmentation pipeline
- Added progressive augmentation scheduler
- Fixed data loading bottleneck"

# Push to GitHub
git push origin main
```

---

## Commit Message Examples

Throughout the project, use clear commit messages:

### Phase 1 Completion:
```
git commit -m "feat: Phase 1 training complete - DeepLabV3+ ResNet-101

Results:
- F1 Score: 0.9087 on test set
- Training time: 16 hours
- Best epoch: 87 (val_f1=0.9087)

Key improvements:
- Enhanced augmentation (ElasticTransform, GridDistortion)
- Progressive augmentation scheduling
- Mixed precision training

Compared to baseline:
- U-Net baseline: 0.9220 F1
- Phase 1 result: 0.9087 F1 (with larger model)"
```

### Phase 2 Comparison:
```
git commit -m "feat: Phase 2 SegFormer-B3 training and evaluation

Results:
- F1 Score: 0.9287 on test set (+2% vs Phase 1)
- Inference: 80 FPS (vs 50 FPS for ResNet-101)
- GPU RAM: 6 GB (same as Phase 1)
- Training time: 18 hours

Model comparison added:
- DeepLabV3+ ResNet-101: 0.9087 F1
- SegFormer-B3: 0.9287 F1 (WINNER)
- Speed & accuracy trade-offs documented"
```

### Phase 3 Sub-phases:
```
git commit -m "feat: Phase 3A SimCLR pretraining complete

Self-supervised learning results:
- Pretraining on 3,432 images (labeled + IR)
- Epochs: 100
- Transfer to segmentation: F1 0.9487 (+2% gain)
- Unlabeled IR images significantly helped

Phase 3B domain adversarial next...
Phase 3C ensemble methods in parallel"
```

---

## Important Notes

### About Your Data/Models (Won't be tracked)
- ❌ `data/raw/` - Raw dataset (11 GB) - Download separately
- ❌ `data/processed/` - Generated splits - Recreate with script
- ❌ `models/*/` - Checkpoints (.pth files) - Train locally
- ❌ `results/` - Training outputs - Generate locally

The `.gitignore` prevents these from being tracked (they're too large).

### What IS tracked (Your code):
- ✅ All `.py` scripts
- ✅ All `.yaml` configs
- ✅ All `.md` documentation
- ✅ Requirements.txt
- ✅ Jupyter notebooks
- ✅ This setup guide

### Cloning on Another Machine:
```powershell
git clone https://github.com/Novabricot/tear-meniscus-segmentation.git
cd tear-meniscus-segmentation

# Install dependencies
pip install -r requirements.txt

# Download dataset separately (not in repo):
# 1. Visit Figshare link in README
# 2. Extract to data/raw/
# 3. Run: python scripts/preprocess_data.py
```

---

## GitHub Features to Use

### 1. Issues (for tracking work)
```powershell
gh issue create --title "Phase 1 training in progress" \
  --body "Started DeepLabV3+ ResNet-101 training on GPU. Expected completion: Week 2."
```

### 2. Projects (for organization)
- Go to: https://github.com/Novabricot/tear-meniscus-segmentation/projects
- Create "Phase 1", "Phase 2", "Phase 3" boards
- Add issues to each phase

### 3. Milestones (for timeline)
- Go to: https://github.com/Novabricot/tear-meniscus-segmentation/milestones
- Create: "Phase 1 (Week 1-2)", "Phase 2 (Week 3-4)", etc.

### 4. Release Tags (after each phase)
```powershell
git tag -a v0.1-phase1 -m "Phase 1 complete: F1=0.9087"
git push origin v0.1-phase1

git tag -a v0.2-phase2 -m "Phase 2 complete: F1=0.9287"
git push origin v0.2-phase2

git tag -a v1.0-final -m "All phases complete: F1=0.94+"
git push origin v1.0-final
```

---

## Troubleshooting

### If `gh auth login` doesn't work:
**Use HTTPS with Personal Access Token instead:**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Check "repo" scope
4. Copy token
5. Run:
   ```powershell
   git config --global credential.helper manager
   git remote add origin https://github.com/Novabricot/tear-meniscus-segmentation.git
   git push -u origin main
   # It will prompt for username (Novabricot) and token (paste it)
   ```

### If "fatal: remote origin already exists":
```powershell
git remote -v  # Check what's there
git remote remove origin  # Remove old
git remote add origin https://github.com/Novabricot/tear-meniscus-segmentation.git
git push -u origin main
```

### If Windows path too long:
Enable long path support:
```powershell
git config --system core.longpaths true
```

---

## Success Checklist

After completing all steps:

- [ ] Git installed and working (`git --version` shows version)
- [ ] GitHub CLI installed (`gh --version` works)
- [ ] Authenticated with GitHub (`gh auth status` shows logged in)
- [ ] Repository created on GitHub
- [ ] Initial commit pushed (can see files at github.com/Novabricot/tear-meniscus-segmentation)
- [ ] README.md displays correctly on GitHub homepage
- [ ] All folders visible (PHASE_1_QUICKWINS, PHASE_2_TRANSFORMERS, etc.)
- [ ] Can create issues and projects

---

## Next Steps

1. ✅ **Install Git** (Steps 1-2)
2. ✅ **Authenticate with GitHub** (Step 3)
3. ✅ **Push project** (Step 4)
4. 📊 **Extract dataset** to `data/raw/` (start Phase 1)
5. 🚀 **Begin Phase 1 training** (Week 1-2)
6. 📝 **Update IMPLEMENTATION_NOTES.md** weekly
7. 📤 **Push progress** after each phase

---

**Questions?**
- GitHub Docs: https://docs.github.com/
- Git Help: https://git-scm.com/book
- GitHub CLI: https://cli.github.com/manual

**Estimated Time**: 10-15 minutes for complete setup  
**Difficulty**: Easy (mostly copy-paste commands)

---

**Created**: May 25, 2026  
**Ready to begin**: Week 1, Phase 1 training
