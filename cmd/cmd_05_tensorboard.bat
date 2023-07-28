@echo off
setlocal

REM call conda activate solveremf2
call conda activate base
cd ..
set "cwd=%CD%"
echo Current working directory: %cwd%

REM tensorboard --logdir /doc/_tensorboard_logs/
jupyter notebook
PAUSE

endlocal