# vector_query_profiler.py


# ==========================================================
# AI Powered Vector Database Query Profiler
# Google Colab / VS Code Compatible
# ==========================================================

# INSTALL REQUIRED PACKAGES FIRST:
# pip install chromadb sentence-transformers pandas gradio psutil pynvml openpyxl torch

import os
import time
import sqlite3
# Optional dependencies are imported safely so the file can be
# analyzed/edited without requiring all packages to be installed.
import importlib

try:
    pd = importlib.import_module("pandas")
except Exception:
    pd = None
if pd is None:
    raise ImportError(
        "Pandas is not installed. Run: pip install pandas"
    )
try:
    psutil = importlib.import_module("psutil")
except Exception:
    psutil = None

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    chromadb = importlib.import_module("chromadb")
except Exception:
    chromadb = None

try:
    torch = importlib.import_module("torch")
    HAS_GPU = getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
except Exception:
    torch = None
    HAS_GPU = False

from concurrent.futures import ThreadPoolExecutor

try:
    st_mod = importlib.import_module("sentence_transformers")
    SentenceTransformer = getattr(st_mod, "SentenceTransformer", None)
except Exception:
    SentenceTransformer = None

# ==========================================================
# GPU DETECTION
# ==========================================================

GPU_HANDLE = None
_pynvml = None

# Use a namespaced import for NVML and guard its usage so static
# analysis doesn't report undefined names when the package isn't present.
if HAS_GPU:
    try:
        _pynvml = importlib.import_module("pynvml")

        _pynvml.nvmlInit()
        GPU_HANDLE = _pynvml.nvmlDeviceGetHandleByIndex(0)

    except Exception:
        _pynvml = None
        HAS_GPU = False

# ==========================================================
# PROJECT FOLDERS
# ==========================================================

os.makedirs("database", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ==========================================================
# DATABASE
# ==========================================================

DB_PATH = "database/query_profiler.db"

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

cursor = conn.cursor()

try:
    cursor.execute("DROP TABLE IF EXISTS query_logs")
except:
    pass

cursor.execute("""
CREATE TABLE IF NOT EXISTS query_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_type TEXT,
    sql_execution_time REAL,
    embedding_time REAL,
    vector_execution_time REAL,
    cpu_usage REAL,
    ram_usage REAL,
    gpu_usage REAL,
    gpu_memory REAL,
    timestamp TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS mock_employees(
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary INTEGER
)
""")

cursor.executemany("""
INSERT OR IGNORE INTO mock_employees
VALUES(?,?,?,?)
""",
[
    (1,"Alice","Engineering",90000),
    (2,"Bob","Data Science",110000),
    (3,"Charlie","DevOps",85000)
])

conn.commit()
# ==========================================================
# VECTOR DATABASE
# ==========================================================

client = None
collection = None

if chromadb is not None:
    try:
        client = chromadb.Client()

        try:
            collection = client.get_collection(
                "documents"
            )
        except Exception:
            collection = client.create_collection(
                "documents"
            )

        try:
            if collection.count() == 0:
                collection.add(
                    documents=[
                        "Database indexing improves query speed",
                        "Vector databases store embeddings",
                        "SQL is used for relational databases",
                        "Employees receive monthly salaries"
                    ],
                    ids=["1","2","3","4"]
                )
        except Exception:
            # if count/add not supported, ignore and continue
            pass

    except Exception:
        client = None
        collection = None

# ==========================================================
# MODEL
# ==========================================================


DEVICE = "cuda" if HAS_GPU else "cpu"

print("Using Device:", DEVICE)

# <-- INSERT PACKAGE CHECK HERE (OPTIONAL)

model = None

model = None

if SentenceTransformer is None:
    raise ImportError(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )

try:
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=DEVICE
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load embedding model: {e}"
    )
    
# simple in-memory embedding cache to avoid repeated encodings
embedding_cache = {}

def optimize_model(use_fp16: bool = False):
    """Put model into eval mode, disable gradients and enable CUDA optimizations."""
    if model is None:
        return

    try:
        model.eval()
    except Exception:
        pass

    if "torch" in globals() and torch is not None:
        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

        if HAS_GPU:
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        if use_fp16:
            try:
                # SentenceTransformers model may support half precision
                model.half()
            except Exception:
                pass

# Run lightweight optimizations at startup
optimize_model()

# ==========================================================
# HARDWARE MONITOR
# ==========================================================

def get_hardware():

    if psutil is None:
        return 0.0, 0.0, 0.0, 0.0

    try:
        cpu = psutil.cpu_percent()
        ram = round(
            psutil.virtual_memory().percent,
            2
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

    gpu = 0
    gpu_mem = 0

    if HAS_GPU:

        try:

            if _pynvml is not None:
                mem = _pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                util = _pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)

                gpu = util.gpu

                gpu_mem = round(
                    mem.used/(1024**2),
                    2
                )
            else:
                gpu = 0
                gpu_mem = 0

        except:
            pass

    return cpu,ram,gpu,gpu_mem

# ==========================================================
# SQL ERROR SUGGESTION
# ==========================================================

def suggest_fix(error):

    error = error.lower()

    if "no such table" in error:
        return "Verify table name"

    if "no such column" in error:
        return "Verify column name"

    if "syntax error" in error:
        return "Check SQL syntax"

    return "Review query"

# ==========================================================
# SQL ENGINE
# ==========================================================

def run_sql(query):

    start = time.time()

    try:

        worker_conn = sqlite3.connect(DB_PATH)

        result = pd.read_sql_query(
         query,
         worker_conn
       )

        worker_conn.close()

        status = "Success"

    except Exception as e:

        result = pd.DataFrame({

            "Error":[str(e)],
            "Suggestion":[
                suggest_fix(str(e))
            ]
        })

        status = "Failed"

    end = time.time()

    return result,end-start,status

# ==========================================================
# VECTOR SEARCH
# ==========================================================

def run_vector_search(text):

    if not text or not text.strip():
        return {}, 0, 0
    if collection is None:
        return{"error": "Vector database not initialized"}, 0, 0

    embed_start = time.time()

    # Use cached embedding when possible
    embedding = embedding_cache.get(text)

    if embedding is None:
        if model is None:
            # fallback: no model available
            embedding = []
        else:
            try:
                # Prefer returning numpy arrays for speed when supported
                embedding = model.encode(
                    [text],
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            except TypeError:
                # older APIs might not support kwargs
                embedding = model.encode([text])
            except Exception:
                embedding = model.encode([text])

        embedding_cache[text] = embedding

    embed_end = time.time()

    embedding_time = embed_end - embed_start

    search_start = time.time()

    # Use text-based query to remain compatible with chromadb clients
    try:
        results = collection.query(
            query_texts=[text],
            n_results=3
        )
    except Exception as e:
        results = {
        "error": str(e)
    }

    search_end = time.time()

    search_time = search_end - search_start

    return results, embedding_time, search_time

# ==========================================================
# MAIN PROFILER
# ==========================================================

def profiler(sql_query,
             vector_query):

    cpu_before,ram_before,\
    gpu_before,gpu_mem_before = get_hardware()

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        sql_future = executor.submit(
            run_sql,
            sql_query
        )

        vector_future = executor.submit(
            run_vector_search,
            vector_query
        )

        sql_df,\
        sql_time,\
        status = sql_future.result()

        vector_results,\
        embedding_time,\
        vector_time = \
        vector_future.result()

    cpu,ram,gpu,gpu_mem = \
        get_hardware()

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
    INSERT INTO query_logs(
        query_type,
        sql_execution_time,
        embedding_time,
        vector_execution_time,
        cpu_usage,
        ram_usage,
        gpu_usage,
        gpu_memory,
        timestamp
    )
    VALUES(?,?,?,?,?,?,?,?,?)
    """,
    (
        status,
        sql_time,
        embedding_time,
        vector_time,
        cpu,
        ram,
        gpu,
        gpu_mem,
        timestamp
    ))

    conn.commit()

    metrics = pd.DataFrame({

        "Metric":[
            "Query Time",
            "Embedding Time",
            "Vector Search Time",
            "CPU Usage",
            "RAM Usage",
            "GPU Usage",
            "GPU Memory"
        ],

        "Value":[
            f"{sql_time:.6f} sec",
            f"{embedding_time:.6f} sec",
            f"{vector_time:.6f} sec",
            f"{cpu} %",
            f"{ram} %",
            f"{gpu} %",
            f"{gpu_mem} MB"
        ]
    })

    return (
        sql_df,
        vector_results,
        metrics
    )

# ==========================================================
# FILE UPLOAD
# ==========================================================

def upload_file(file):

    if file is None:

        return pd.DataFrame({
            "Status":[
                "No file selected"
            ]
        })

    try:

        if file.name.endswith(
            ".csv"
        ):

            df = pd.read_csv(
                file.name
            )

        elif file.name.endswith(
            ".xlsx"
        ):

            df = pd.read_excel(
                file.name
            )

        else:

            return pd.DataFrame({

                "Error":[
                    "Unsupported file"
                ]
            })

        df.to_sql(
            "uploaded_data",
            conn,
            if_exists="replace",
            index=False
        )

        return df.head(10)

    except Exception as e:

        return pd.DataFrame({

            "Error":[str(e)]
        })

# ==========================================================
# SCHEMA VIEWER
# ==========================================================

def show_schema():

    return pd.read_sql_query("""

    SELECT name

    FROM sqlite_master

    WHERE type='table'

    """,conn)

# ==========================================================
# TABLE PREVIEW
# ==========================================================

def preview_table(table):

    try:

        return pd.read_sql_query(

            f"SELECT * FROM {table} LIMIT 20",

            conn

        )

    except Exception as e:

        return pd.DataFrame({

            "Error":[str(e)]

        })

# ==========================================================
# LOG HISTORY
# ==========================================================

def get_logs():

    return pd.read_sql_query(

        "SELECT * FROM query_logs",

        conn

    )

# ==========================================================
# EXPORT LOGS
# ==========================================================

def export_logs():

    filename = "query_logs.csv"

    get_logs().to_csv(
        filename,
        index=False
    )

    return filename
# ===================== TEST BLOCK =====================

print("Testing profiler...")

try:
    a, b, c = profiler(
        "SELECT * FROM mock_employees",
        "salary information"
    )

    print("SQL OK")
    print(a)

    print("VECTOR OK")
    print(b)

    print("METRICS OK")
    print(c)

except Exception as e:
    import traceback
    traceback.print_exc()

# ==========================================================
# GRADIO UI
# ==========================================================

if gr is not None:
    with gr.Blocks() as app:

        gr.Markdown("# Vector Database Query Profiler")

        with gr.Tab("Profiler"):

            sql_box = gr.Textbox(value="SELECT * FROM mock_employees")
            vector_box = gr.Textbox(value="salary information")

            run_btn = gr.Button("Run Profiler")

            sql_output = gr.Dataframe()
            vector_output = gr.JSON()
            metric_output = gr.Dataframe()

            def safe_profiler(sql_query, vector_query):
                try:
                    return profiler(sql_query, vector_query)
                except Exception as e:
                    import traceback
                    traceback.print_exc()

                    return (
                        pd.DataFrame({"Error":[str(e)]}),
                        {"error": str(e)},
                        pd.DataFrame({"Error":[str(e)]})
                    )

            run_btn.click(
                safe_profiler,
                [sql_box, vector_box],
                [sql_output, vector_output, metric_output]
            )

        with gr.Tab("Upload"):
            uploader = gr.File()
            upload_view = gr.Dataframe()

            uploader.upload(
                upload_file,
                uploader,
                upload_view
            )

        with gr.Tab("Schema"):
            schema_btn = gr.Button("Show Tables")
            schema_view = gr.Dataframe()

            schema_btn.click(
                show_schema,
                outputs=schema_view
            )

        with gr.Tab("History"):
            history_btn = gr.Button("Refresh")
            history_view = gr.Dataframe()

            history_btn.click(
                get_logs,
                outputs=history_view
            )

            export_btn = gr.Button("Export Logs")
            export_file = gr.File()

            export_btn.click(
                export_logs,
                outputs=export_file
            )
try:
    app.launch(
        share=False,
        debug=True,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )

except Exception:
    print("Gradio UI failed to start; continuing without UI.")
else:
    print("Gradio not available; skipping UI. You can still call profiler() programmatically.")

