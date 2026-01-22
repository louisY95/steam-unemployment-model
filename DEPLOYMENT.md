# 🚀 GitHub Actions Deployment Guide

This guide will help you set up automated data collection using GitHub Actions (completely free, no local PC needed!).

## ✅ Prerequisites

- GitHub account (which you have!)
- FRED API key: `7285d44800d2ba421ac2017dcdded78f`

## 📋 Step-by-Step Setup

### 1. Create GitHub Repository

```bash
# Navigate to your project directory
cd c:\Users\louis\Desktop\steam_unemployment_model

# Initialize git (already done)
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Steam unemployment prediction model"
```

### 2. Create GitHub Repository Online

1. Go to https://github.com/new
2. Repository name: `steam-unemployment-model` (or any name you prefer)
3. Description: "Predictive model analyzing Steam user activity vs US unemployment rates"
4. Choose **Public** (for free GitHub Actions minutes) or **Private**
5. Do NOT initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

### 3. Push Code to GitHub

GitHub will show you commands like these. Run them:

```bash
git remote add origin https://github.com/YOUR_USERNAME/steam-unemployment-model.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### 4. Add FRED API Key as GitHub Secret

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. In left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `FRED_API_KEY`
6. Value: `7285d44800d2ba421ac2017dcdded78f`
7. Click **Add secret**

### 5. Enable GitHub Actions

1. Go to the **Actions** tab in your repository
2. If prompted, click **"I understand my workflows, go ahead and enable them"**
3. You should see the workflow "Collect Steam & Unemployment Data"

### 6. Test Manual Run

1. In the **Actions** tab, click on "Collect Steam & Unemployment Data"
2. Click **Run workflow** (button on the right)
3. Check "Also collect FRED data"
4. Click **Run workflow**
5. Watch it run! (takes ~2-3 minutes)

### 7. Verify It's Working

After the workflow completes:

1. Check the **Summary** page - should show ✅ for data collection
2. Go to **Actions** → Click on the run → **Artifacts** section
3. Download `steam-data` artifact to verify data was collected
4. The workflow will now run **automatically every hour**!

## 🔄 How It Works

- **Hourly Collection**: Runs every hour to collect Steam data
- **Weekly FRED Update**: Collects unemployment data every Sunday at midnight
- **Data Storage**:
  - Artifacts stored for 90 days
  - Also committed to git (if repo is private)
- **Cost**: **100% FREE** (GitHub gives 2,000 minutes/month for public repos, ~20-30 minutes per day for this)

## 📊 Monitoring

### View Collection Status

1. Go to **Actions** tab
2. Click on latest workflow run
3. View logs and summary

### Download Collected Data

1. **Actions** tab → Click a workflow run
2. Scroll down to **Artifacts**
3. Download `steam-data`
4. Extract and place files in your local `data/raw/` folder

### Manual Trigger

Anytime you want to force a collection:
1. **Actions** → "Collect Steam & Unemployment Data"
2. **Run workflow**
3. Optionally check "Also collect FRED data"

## 🛠️ Troubleshooting

### Workflow Fails

- Check the logs in the Actions tab
- Most common issue: FRED API key not set correctly
- Solution: Re-add the secret in Settings → Secrets

### Want to Change Collection Frequency?

Edit `.github/workflows/collect_data.yml`:
```yaml
schedule:
  - cron: '0 */2 * * *'  # Every 2 hours
  - cron: '*/30 * * * *'  # Every 30 minutes
  - cron: '0 0 * * *'     # Once per day at midnight
```

## 🎉 You're Done!

The system will now:
- ✅ Collect Steam data every hour
- ✅ Collect unemployment data weekly
- ✅ Store data in GitHub
- ✅ Run 24/7 without your PC

After 30-60 days, you'll have enough historical data to run the full statistical analysis!

## 📥 Running Analysis Later

Once you have enough data (30+ days):

```bash
# Download data from GitHub artifacts
# Place in data/raw/ folder

# Process the data
python main.py process

# Run analysis
python main.py analyze

# Generate report
python main.py report
```
