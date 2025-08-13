@echo off
REM Windows batch file for starting AI-Powered Chatbot System

echo ====================================
echo AI-Powered Chatbot System Launcher
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected

REM Parse command line arguments
set SKIP_INSTALL=0
set INSTALL_ONLY=0
set BACKEND_ONLY=0

:parse_args
if "%1"=="" goto end_parse
if "%1"=="--skip-install" set SKIP_INSTALL=1
if "%1"=="--install-only" set INSTALL_ONLY=1
if "%1"=="--backend-only" set BACKEND_ONLY=1
if "%1"=="--help" goto show_help
shift
goto parse_args
:end_parse

REM Create virtual environment if it doesn't exist
if not exist "venv" if %SKIP_INSTALL%==0 (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
if exist "venv" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install dependencies
if %SKIP_INSTALL%==0 (
    echo.
    echo Installing dependencies...
    
    REM Upgrade pip
    python -m pip install --upgrade pip
    
    REM Install backend dependencies
    if exist "backend\requirements.txt" (
        pip install -r backend\requirements.txt
        echo Backend dependencies installed
    ) else (
        echo WARNING: backend\requirements.txt not found
    )
    
    REM Download NLTK data
    echo Downloading NLTK data...
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" 2>nul
    
    REM Download spaCy model
    echo Downloading spaCy model...
    python -m spacy download en_core_web_sm 2>nul
)

REM Exit if install-only mode
if %INSTALL_ONLY%==1 (
    echo Installation complete
    pause
    exit /b 0
)

REM Create necessary directories
echo.
echo Creating directories...
if not exist "backend\data" mkdir "backend\data"
if not exist "backend\models" mkdir "backend\models"
if not exist "backend\checkpoints" mkdir "backend\checkpoints"
if not exist "backend\logs" mkdir "backend\logs"
if not exist "backend\domain_knowledge" mkdir "backend\domain_knowledge"
if not exist "backend\faiss_index" mkdir "backend\faiss_index"
if not exist "backend\chroma_db" mkdir "backend\chroma_db"
if not exist "backend\knowledge_base" mkdir "backend\knowledge_base"
if not exist "frontend" mkdir "frontend"
if not exist "logs" mkdir "logs"

REM Check if ports are available
echo.
echo Checking ports...
netstat -an | findstr :8000 >nul
if %errorlevel%==0 (
    echo WARNING: Port 8000 is already in use
    choice /M "Continue anyway?"
    if errorlevel 2 exit /b 1
)

netstat -an | findstr :8001 >nul
if %errorlevel%==0 (
    echo WARNING: Port 8001 is already in use
    choice /M "Continue anyway?"
    if errorlevel 2 exit /b 1
)

netstat -an | findstr :3000 >nul
if %errorlevel%==0 (
    echo WARNING: Port 3000 is already in use
    choice /M "Continue anyway?"
    if errorlevel 2 exit /b 1
)

REM Start backend services
echo.
echo Starting backend services...

REM Start main API
echo Starting main API on port 8000...
cd backend
start /b python main.py > ..\logs\main_api.log 2>&1
cd ..
timeout /t 2 /nobreak >nul

REM Start training API
echo Starting training API on port 8001...
cd backend
start /b python training_interface.py > ..\logs\training_api.log 2>&1
cd ..
timeout /t 2 /nobreak >nul

REM Start frontend if not backend-only
if %BACKEND_ONLY%==0 (
    echo.
    echo Starting frontend...
    
    REM Check if frontend exists
    if exist "frontend" (
        cd frontend
        
        REM Check for package.json and npm
        if exist "package.json" (
            where npm >nul 2>&1
            if %errorlevel%==0 (
                call npm install
                start /b npm start > ..\logs\frontend.log 2>&1
                echo Frontend started
            ) else (
                goto start_simple_server
            )
        ) else (
            :start_simple_server
            REM Create basic frontend if it doesn't exist
            if not exist "index.html" (
                cd ..
                python -c "from start_system import SystemManager; manager = SystemManager(); manager.create_basic_frontend()"
                cd frontend
            )
            
            REM Start simple HTTP server
            start /b python -m http.server 3000 > ..\logs\frontend.log 2>&1
            echo Frontend started (basic mode)
        )
        
        cd ..
    )
)

REM Wait for services to start
timeout /t 3 /nobreak >nul

REM Print access information
cls
echo ========================================
echo     AI-Powered Chatbot System
echo          System is ready!
echo ========================================
echo.
echo Access your services at:
echo   Frontend:        http://localhost:3000
echo   Main API:        http://localhost:8000
echo   Training API:    http://localhost:8001
echo.
echo Demo interfaces:
echo   API Docs:        http://localhost:8000/docs
echo   Voice Demo:      http://localhost:8000/voice_demo.html
echo   Automation:      http://localhost:8000/automation_demo.html
echo   RAG System:      http://localhost:8000/rag_demo.html
echo   LLM Training:    http://localhost:8001/llm_demo.html
echo.
echo To stop all services, run: stop_system.bat
echo.

REM Open browser
timeout /t 2 /nobreak >nul
start http://localhost:3000

REM Create stop script
echo @echo off > stop_system.bat
echo echo Stopping AI-Powered Chatbot System... >> stop_system.bat
echo taskkill /F /IM python.exe 2^>nul >> stop_system.bat
echo taskkill /F /FI "WINDOWTITLE eq npm*" 2^>nul >> stop_system.bat
echo echo All services stopped >> stop_system.bat
echo pause >> stop_system.bat

echo Press any key to keep services running...
pause >nul
goto :eof

:show_help
echo Usage: %0 [options]
echo.
echo Options:
echo   --skip-install    Skip dependency installation
echo   --install-only    Only install dependencies, don't start services
echo   --backend-only    Only start backend services
echo   --help           Show this help message
pause
exit /b 0