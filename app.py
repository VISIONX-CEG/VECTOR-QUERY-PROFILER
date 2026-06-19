from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# import your existing functions
from import_os_1 import profiler, show_schema, get_logs, upload_file

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows HTML to talk to backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/profiler")
def run(data: dict):
    sql = data["sql"]
    vector = data["vector"]

    sql_res, vec_res, metrics = profiler(sql, vector)

    return {
        "sql_result": sql_res.to_dict(),
        "vector_result": vec_res,
        "metrics": metrics.to_dict()
    }

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    return upload_file(file.file)

@app.get("/schema")
def schema():
    return show_schema().to_dict()

@app.get("/history")
def history():
    return get_logs().to_dict()