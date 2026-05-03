@echo off
REM Launch the LawMate Streamlit demo.
REM On first run, also creates .venv, installs deps, and builds the FAISS index.

setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [setup] .venv not found - bootstrapping with uv...
  uv venv .venv --python 3.11 || goto :err
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :err
)

if not exist "experiments\index\kb.faiss" (
  echo [setup] FAISS index missing - building from data\knowledge_base\knowledge_base.csv...
  .venv\Scripts\python.exe scripts\build_index.py || goto :err
)

echo [run] Starting Streamlit demo at http://localhost:8501 ...
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
exit /b 0

:err
echo.
echo [error] Setup step failed. Make sure 'uv' is on PATH (installer puts it at C:\Users\%%USERNAME%%\.local\bin\uv.exe).
exit /b 1
