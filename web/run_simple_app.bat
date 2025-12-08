@echo off
echo ========================================
echo    SIMPLE APP - LAPTOP RECOMMENDATION
echo ========================================
echo.

echo Installing packages...
pip install streamlit requests

echo.
echo Starting Simple App...
streamlit run simple_app.py

pause
