@echo off
echo Starting Interactive Burst Detection Demo...
echo.
echo Navigate to the URL shown below once the server starts
echo Press Ctrl+C to stop the server
echo.
cd /d D:\GitHub\cib-mango-tree-peter
python -m streamlit run demo\interactive_burst_app.py --server.port 8503

