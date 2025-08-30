#!/usr/bin/env python3
"""
Minimal test to check if FastAPI server can start
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting minimal test server on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)