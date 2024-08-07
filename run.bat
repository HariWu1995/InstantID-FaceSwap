@echo off

set GIT=
set PYTHON="C:\Program Files\Python310\python.exe"
set VENV_DIR=C:\Users\Mr. RIAH\Documents\GenAI\sd_env

::TIMEOUT /T 1

call %PYTHON% -m pip install accelerate peft insightface controlnet_aux opencv-python
call %PYTHON% webui/app_faceswap.py

pause
