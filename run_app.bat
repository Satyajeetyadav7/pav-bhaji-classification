@echo off
REM Launch the pav bhaji upload UI. Trains the models first if they are missing.
cd /d "%~dp0"
if not exist "models\image_model.joblib" (
    echo Training models, this takes a couple of minutes...
    python app\train_models.py
)
python -m streamlit run app\app.py
