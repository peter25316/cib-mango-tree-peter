@echo off
echo Starting Interactive Burst Detection v2...
echo.
echo This version has working burst region clicking!
echo Click on colored burst circles to explore contributors.
echo.
python -m streamlit run demo/interactive_burst_app_v2.py --server.port 8504
pause

