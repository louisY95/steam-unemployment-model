# 🚀 Quick Start Guide

## Option 1: Automated Setup (Easiest!)

Just run this command in your terminal:

```bash
.\setup_github.bat
```

Follow the prompts and you're done!

## Option 2: Manual Setup

### 1. Create GitHub Repository

Go to https://github.com/new and create a new repository:
- Name: `steam-unemployment-model`
- Public (for free Actions)
- Don't initialize with anything

### 2. Push Your Code

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/steam-unemployment-model.git
git push -u origin main
```

### 3. Add API Key Secret

1. Go to your repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `FRED_API_KEY`
4. Value: `7285d44800d2ba421ac2017dcdded78f`
5. Click "Add secret"

### 4. Enable and Test Workflow

1. Go to the **Actions** tab
2. Enable workflows if prompted
3. Click "Collect Steam & Unemployment Data"
4. Click "Run workflow" → Check "Also collect FRED data" → Run
5. Watch it collect data!

## ✅ You're Done!

The system will now automatically:
- Collect Steam data **every hour**
- Collect unemployment data **every Sunday**
- Store data in GitHub artifacts (90 days retention)
- Run 24/7 without needing your PC on

## 📊 What Happens Next?

- **Week 1-4**: Building up historical dataset
- **Week 4-8**: Accumulating more data for better analysis
- **After 30 days**: Enough data for initial analysis
- **After 60 days**: Robust dataset for full statistical analysis

## 🔍 Monitor Your Collection

Check progress anytime:
1. Go to your GitHub repo
2. Click **Actions** tab
3. See all collection runs and their status

## 📥 Download Collected Data

When ready to analyze:
1. Actions → Click any workflow run
2. Scroll to **Artifacts** section
3. Download `steam-data` ZIP file
4. Extract to your local `data/` folder
5. Run: `python main.py process && python main.py analyze`

## 💡 Tips

- **First run might fail**: That's OK! It's setting up. Try running manually once.
- **Check logs**: If collection fails, view the workflow logs to see why
- **Cost**: Completely FREE for public repositories
- **Change frequency**: Edit `.github/workflows/collect_data.yml` cron schedule

## 🆘 Need Help?

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions and troubleshooting.

---

**Remember**: No local PC needed once this is set up. GitHub servers do all the work! 🎉
