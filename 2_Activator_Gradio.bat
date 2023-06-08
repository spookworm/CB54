@echo off

call conda activate solveremf2
REM python app.py runserver 127.0.0.1:7860
REM TaskKill /PID 7860 /F
gradio app.py