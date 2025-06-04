@echo off
echo Starting Dental Diagnosis Web App...
cd /d %~dp0
python -m streamlit run streamlit_app.py
pause