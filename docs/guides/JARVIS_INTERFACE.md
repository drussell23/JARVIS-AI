# 🎯 JARVIS Iron Man Interface

## Overview
The JARVIS interface is a React-based frontend that provides an Iron Man-inspired UI for interacting with your AI assistant. It features a futuristic, holographic design with cyan/blue color schemes, glass-morphism effects, and HUD-style elements.

## Features
- **Iron Man-inspired Design**: Futuristic UI with holographic effects
- **Real-time Chat**: Seamless communication with Claude API
- **Responsive Layout**: Works on desktop and mobile devices
- **HUD Elements**: Scanlines, grid patterns, and status indicators
- **Custom Fonts**: Orbitron and Rajdhani for that sci-fi feel

## Starting the Interface

### Method 1: Using start_system.py (Recommended)
```bash
python3 start_system.py
```

The script will:
1. Start the backend API (with Claude integration)
2. Start the JARVIS React interface on port 3000
3. Automatically open the interface in your browser

### Method 2: Manual Start
```bash
# Terminal 1: Start backend
cd backend
python3 run_server.py

# Terminal 2: Start frontend
cd frontend
npm start
```

## Accessing the Interface
- **JARVIS Interface**: http://localhost:3000/ (Iron Man UI)
- **API Documentation**: http://localhost:8000/docs
- **Basic Chat**: http://localhost:8000/

## First Time Setup
If you haven't installed the frontend dependencies yet:
```bash
cd frontend
npm install
```

## Troubleshooting

### Port Already in Use
If port 3000 is already in use, the start script will notify you. You can either:
1. Stop the process using port 3000
2. Change the port in the environment: `PORT=3001 npm start`

### React Takes Time to Compile
The React development server may take 10-15 seconds to compile on first start. The browser will auto-refresh when ready.

### API Connection Issues
Make sure the backend is running on port 8000. The frontend expects the API at `http://127.0.0.1:8000/chat`.

## Customization
- **Colors**: Edit `/frontend/src/App.css` CSS variables
- **Layout**: Modify `/frontend/src/App.js`
- **API Endpoint**: Update the axios URL in `App.js` if needed

## Claude Integration
The interface automatically uses Claude API when configured. Make sure your `.env` file contains:
```env
ANTHROPIC_API_KEY=your-key-here
USE_CLAUDE=1
```

Enjoy your Iron Man-inspired AI assistant! 🚀