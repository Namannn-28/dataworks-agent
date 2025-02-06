from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional, Dict, Any
import httpx
import shutil
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

class TaskProcessor:
    def __init__(self):
        self.llm_token = os.environ.get("AIPROXY_TOKEN")
        if not self.llm_token:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
        
        # Initialize paths
        self.data_dir = Path("/data")
        
    async def call_llm(self, prompt: str) -> str:
        """Call the LLM (GPT-4o-Mini) through AI Proxy"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.aiproxy.xyz/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "prompt": prompt,
                    "max_tokens": 500
                }
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="LLM API call failed")
            return response.json()["choices"][0]["text"].strip()

    async def extract_task_type(self, task_description: str) -> Dict[str, Any]:
        """Use LLM to analyze the task and extract key information"""
        prompt = f"""Analyze this task and extract key information:
Task: {task_description}
Return a JSON with these fields:
- task_type: The type of task (A1-A10 or B1-B10)
- input_path: Input file path if any
- output_path: Output file path if any
- operation: Basic operation description
"""
        result = await self.call_llm(prompt)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse task")

    async def process_task(self, task_description: str) -> dict:
        """Main task processing logic"""
        task_info = await self.extract_task_type(task_description)
        
        # Security check: Ensure all paths are within /data
        if task_info.get('input_path') and not str(task_info['input_path']).startswith('/data/'):
            raise HTTPException(status_code=400, detail="Cannot access files outside /data directory")
        if task_info.get('output_path') and not str(task_info['output_path']).startswith('/data/'):
            raise HTTPException(status_code=400, detail="Cannot write files outside /data directory")

        try:
            if task_info['task_type'] == 'A1':
                return await self.handle_datagen(task_info)
            elif task_info['task_type'] == 'A2':
                return await self.handle_prettier_format(task_info)
            elif task_info['task_type'] == 'A3':
                return await self.handle_date_counting(task_info)
            elif task_info['task_type'] == 'A4':
                return await self.handle_contact_sorting(task_info)
            elif task_info['task_type'] == 'A5':
                return await self.handle_log_processing(task_info)
            elif task_info['task_type'] == 'A6':
                return await self.handle_markdown_index(task_info)
            elif task_info['task_type'] == 'A7':
                return await self.handle_email_extraction(task_info)
            elif task_info['task_type'] == 'A8':
                return await self.handle_card_extraction(task_info)
            elif task_info['task_type'] == 'A9':
                return await self.handle_similar_comments(task_info)
            elif task_info['task_type'] == 'A10':
                return await self.handle_ticket_sales(task_info)
            else:
                # Handle B-series tasks
                return await self.handle_business_task(task_info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def handle_datagen(self, task_info: dict) -> dict:
        """Handle running the data generation script"""
        try:
            # Install uv if not present
            subprocess.run(["which", "uv"], check=False)
            if subprocess.returncode != 0:
                subprocess.run(["pip", "install", "uv"], check=True)
            
            # Download and run datagen.py
            script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
            async with httpx.AsyncClient() as client:
                response = await client.get(script_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to download script")
                
                with open("datagen.py", "w") as f:
                    f.write(response.text)
                
                # Run the script with email argument
                subprocess.run(["python", "datagen.py", os.environ.get("USER_EMAIL", "default@example.com")], check=True)
                
            return {"status": "success", "message": "Data generation completed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

    async def handle_prettier_format(self, task_info: dict) -> dict:
        """Handle formatting with prettier"""
        try:
            # Install prettier if needed
            subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True)
            
            input_path = task_info['input_path']
            subprocess.run(["prettier", "--write", input_path], check=True)
            
            return {"status": "success", "message": "File formatted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Formatting failed: {str(e)}")

    async def handle_date_counting(self, task_info: dict) -> dict:
        """Count specific weekdays in a file"""
        try:
            input_path = task_info['input_path']
            output_path = task_info['output_path']
            
            with open(input_path, 'r') as f:
                dates = [line.strip() for line in f]
            
            count = sum(1 for date in dates if datetime.strptime(date, '%Y-%m-%d').strftime('%A') == 'Wednesday')
            
            with open(output_path, 'w') as f:
                f.write(str(count))
            
            return {"status": "success", "count": count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Date counting failed: {str(e)}")

    # ... Additional handler methods for other tasks would go here ...

    def safe_path_check(self, path: str) -> bool:
        """Check if a path is safe (within /data directory)"""
        try:
            path = Path(path).resolve()
            return str(path).startswith('/data/')
        except Exception:
            return False

processor = TaskProcessor()

@app.post("/run")
async def run_task(task: str):
    """Execute a task based on plain-English description"""
    try:
        result = await processor.process_task(task)
        return {"status": "success", "result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str):
    """Read and return file contents"""
    if not processor.safe_path_check(path):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
