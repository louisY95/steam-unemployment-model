@echo off
echo ========================================
echo GitHub Setup for Steam Unemployment Model
echo ========================================
echo.

echo Step 1: Creating GitHub repository...
echo.
echo Please go to: https://github.com/new
echo.
echo Repository settings:
echo   Name: steam-unemployment-model
echo   Description: Predictive model for Steam activity vs US unemployment
echo   Visibility: Public (for free GitHub Actions)
echo   Do NOT initialize with README
echo.
pause

echo.
echo Step 2: What is your GitHub username?
set /p GITHUB_USER="Enter your GitHub username: "

echo.
echo Step 3: Adding remote and pushing code...
git remote add origin https://github.com/%GITHUB_USER%/steam-unemployment-model.git
git push -u origin main

echo.
echo ========================================
echo SUCCESS! Code pushed to GitHub
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Go to: https://github.com/%GITHUB_USER%/steam-unemployment-model/settings/secrets/actions
echo.
echo 2. Click "New repository secret"
echo.
echo 3. Add secret:
echo    Name: FRED_API_KEY
echo    Value: 7285d44800d2ba421ac2017dcdded78f
echo.
echo 4. Go to Actions tab and enable workflows
echo.
echo 5. Manually run first collection to test
echo.
echo See DEPLOYMENT.md for full instructions!
echo.
pause
