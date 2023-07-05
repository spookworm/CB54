@echo off

call conda activate solveremf2
REM python solveremf2.py runserver 127.0.0.1:7860
REM TaskKill /PID 7860 /F
cd ..
gradio main.py