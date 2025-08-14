#!/usr/bin/env python3
"""Simple test API to verify basic functionality"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test API")

@app.get("/")
def read_root():
    return {"message": "Test API is working!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting Test API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)