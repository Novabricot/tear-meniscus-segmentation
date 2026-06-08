# GitHub Setup & Push Instructions

Since you need GitHub authentication set up, follow these steps:

## Option 1: SSH Authentication (Recommended)

### Step 1: Generate SSH Key (if you don't have one)
```bash
ssh-keygen -t ed25519 -C "your-github-email@example.com"
# or if ed25519 not supported:
ssh-keygen -t rsa -b 4096 -C "your-github-email@example.com"
```
- Press Enter to save to default location (`~/.ssh/id_ed25519`)
- Enter a passphrase (optional, but recommended for security)

### Step 2: Add SSH Key to SSH Agent
```bash
# Windows PowerShell:
$env:GIT_SSH_COMMAND="ssh -i C:\Users\YOUR_USERNAME\.ssh\id_ed25519"

# Or for all sessions, add to git config:
git config --global core.sshCommand "ssh -i C:\Users\YOUR_USERNAME\.ssh\id_ed25519"
```

### Step 3: Add Public Key to GitHub
1. Copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
2. Go to: https://github.com/settings/keys
3. Click "New SSH key"
4. Paste your public key
5. Save

### Step 4: Test SSH Connection
```bash
ssh -T git@github.com
# Should output: Hi Novabricot! You've successfully authenticated...
```

---

## Option 2: HTTPS with Personal Access Token

### Step 1: Create Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. **Scopes**: Check `repo` (full control of private/public repos)
4. **Expiration**: 90 days or custom
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### Step 2: Configure Git Credential Manager
```bash
# Install Git Credential Manager (Windows)
# Usually already installed with modern Git
git config --global credential.helper manager

# Or use wincred:
git config --global credential.helper wincred
```

### Step 3: Test Authentication
```bash
git clone https://github.com/Novabricot/tear-meniscus-segmentation.git
# It will prompt for username & token
# Username: Novabricot
# Password: (paste your token here)
```

---

## Option 3: GitHub CLI (Easiest)

### Step 1: Install GitHub CLI
```bash
# Windows - using Winget:
winget install GitHub.cli

# Or Chocolatey:
choco install gh

# Or download from: https://cli.github.com
```

### Step 2: Authenticate
```bash
gh auth login
# Choose: GitHub.com
# Choose: HTTPS
# Choose: Yes (Authenticate with browser)
# Browser will open, authorize the app
```

### Step 3: Create Repository
```bash
cd tear-meniscus-segmentation
gh repo create tear-meniscus-segmentation --public --source=. --push
```

**This is the easiest method - recommended!**

---

## Initialize Git & Push to GitHub

### If using Option 1 (SSH) or Option 2 (HTTPS):

```bash
# Navigate to project directory
cd tear-meniscus-segmentation

# Initialize git
git init

# Configure git (if not already done globally)
git config user.name "Your Name"
git config user.email "your-email@example.com"

# Add all files
git add .

# Create initial commit
git commit -m "init: Initial project structure with 3-phase training plan

- Phase 1: DeepLabV3+ ResNet-101 with enhanced augmentation
- Phase 2: SegFormer-B3 transformer-based segmentation
- Phase 3: SimCLR pretraining, domain adversarial, ensemble methods
- Complete documentation and implementation guides
- Training configs for all phases"

# Add remote (replace with SSH if using Option 1)
git remote add origin https://github.com/Novabricot/tear-meniscus-segmentation.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### If using Option 3 (GitHub CLI):
```bash
cd tear-meniscus-segmentation
gh repo create tear-meniscus-segmentation \
  --public \
  --source=. \
  --push \
  --description="Improving tear meniscus segmentation for dry eye detection - 3-phase research project"
```

---

## Verify Push

Check on GitHub:
```
https://github.com/Novabricot/tear-meniscus-segmentation
```

You should see:
- ✅ All folders and files
- ✅ .gitignore configured
- ✅ README.md displayed on main page
- ✅ LICENSE visible
- ✅ Phase guides in docs/

---

## Common Issues & Solutions

### Issue: "Permission denied (publickey)"
- **Cause**: SSH key not set up correctly
- **Solution**: 
  1. Check SSH key exists: `ls ~/.ssh/id_ed25519`
  2. Add to SSH agent: `ssh-add ~/.ssh/id_ed25519`
  3. Test: `ssh -T git@github.com`

### Issue: "fatal: remote origin already exists"
- **Solution**: Remove and re-add
  ```bash
  git remote remove origin
  git remote add origin https://github.com/Novabricot/tear-meniscus-segmentation.git
  ```

### Issue: "fatal: could not read Username"
- **Cause**: Using HTTPS but no credentials configured
- **Solution**: Use GitHub CLI or configure credential manager

### Issue: Large files being tracked
- **Check**: `git ls-files | grep -E "\.pth|\.pt|\.zip|\.tar"` 
- **Fix**: Add to .gitignore and remove from tracking
  ```bash
  git rm --cached models/**/*.pth
  git commit -m "docs: Remove checkpoint files from tracking"
  ```

---

## After Initial Push: Daily Workflow

```bash
# Pull latest changes
git pull origin main

# Make changes to code
# ... edit files ...

# Check what changed
git status

# Stage changes
git add scripts/ configs/  # Only code, not data/models

# Commit with clear message
git commit -m "feat: Implement Phase 1 training script with ResNet-101"

# Push to GitHub
git push origin main
```

---

## What NOT to Commit

The `.gitignore` file already excludes:
- ❌ `data/raw/` (large dataset)
- ❌ `data/processed/` (generated files)
- ❌ `models/**/*.pth` (large checkpoints)
- ❌ `results/` (large outputs)
- ❌ `*.log`, `tensorboard_logs/`, `.wandb/`
- ❌ `.venv/`, `__pycache__/`

**DO commit**:
- ✅ Python scripts (.py files)
- ✅ Configuration files (.yaml)
- ✅ Documentation (.md files)
- ✅ .gitignore, LICENSE, README
- ✅ Requirements.txt
- ✅ Jupyter notebooks (.ipynb)

---

## Helpful GitHub Features

### Create an Issue for tracking
```bash
# Command line (if using GitHub CLI)
gh issue create --title "Phase 1 training in progress" --body "Started training DeepLabV3+ ResNet-101"
```

### Create a Project Board
1. Go to: https://github.com/Novabricot/tear-meniscus-segmentation/projects
2. Click "Create a project"
3. Add cards for each phase

### Add Release Tags (after each phase)
```bash
git tag -a v0.1-phase1-complete -m "Phase 1: DeepLabV3+ ResNet-101 training complete"
git push origin v0.1-phase1-complete
```

---

## Need Help?

- **GitHub Docs**: https://docs.github.com/en/get-started
- **Git Reference**: https://git-scm.com/book/en/v2
- **SSH Setup**: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- **GitHub CLI**: https://cli.github.com/

---

**Recommended**: Use GitHub CLI (Option 3) for easiest setup!
