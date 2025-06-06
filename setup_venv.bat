@echo off
REM Install virtualenv if not already installed
pip install virtualenv

REM Create a virtual environment
virtualenv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Run the application
streamlit run streamlit_app.py