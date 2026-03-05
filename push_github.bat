@echo off
echo ========================================
echo PUSH VERS GITHUB
echo ========================================
echo.

REM Verifier si git est initialise
if not exist .git (
    echo Initialisation de Git...
    git init
    git remote add origin https://github.com/HadilMhadhbi/projet-python.git
)

echo.
echo Ajout des fichiers...
git add .

echo.
echo Commit...
git commit -m "Semaines 2 et 3: Preprocessing + Modeling avec Boosting et MLflow"

echo.
echo Push vers GitHub...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo TERMINE!
echo ========================================
pause
