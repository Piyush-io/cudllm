# FSR CUDA Kernel Generator

Automated CUDA kernel generation using LLM-powered Feedback-driven Search and Refinement (FSR).

## Overview

This project implements an FSR framework that:
- Generates CUDA kernel candidates using LLM (Groq/LangChain)
- Compiles kernels with `nvcc`
- Validates correctness through automated testing
- Profiles performance using CUDA events
- Iteratively refines prompts based on compilation errors, validation failures, and performance metrics
- Uses hierarchical RAG (Retrieval-Augmented Generation) with ChromaDB to provide context-aware knowledge

## Features

- **Interactive Menu Interface** - No CLI flags to remember
- **Cloud-First ChromaDB** - Supports both cloud and local persistent storage
- **Auto GPU Detection** - Automatically detects your GPU architecture
- **Hierarchical RAG** - Context retrieval across 5 knowledge stages
- **Iterative Refinement** - Learns from compilation and performance feedback
- **Clean Temp Management** - Automatic cleanup of compilation artifacts

## Architecture

```
┌─────────────────┐
│  Interactive    │
│     Menu        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  FSR Framework  │◄────►│ Prompt Mgr   │
└────────┬────────┘      └──────┬───────┘
         │                      │
         │                      ▼
         │              ┌──────────────┐
         │              │ Hierarchical │
         │              │     RAG      │
         │              └──────┬───────┘
         │                     │
         ▼                     ▼
┌─────────────────┐    ┌──────────────┐
│  LLM Interface  │    │  ChromaDB    │
│   (Groq API)    │    │   (Cloud)    │
└─────────────────┘    └──────────────┘
         │
         ▼
┌─────────────────┐
│   Execution     │
│   Checkers      │
├─────────────────┤
│ • Compilation   │
│ • Validation    │
│ • Profiling     │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- CUDA Toolkit (with `nvcc`)
- `uv` package manager ([installation](https://github.com/astral-sh/uv))
- ChromaDB Cloud account (or local persistent mode)
- Groq API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd weaver-agent-workflow
   ```

2. **Install dependencies**
   ```bash
   uv sync
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env with your credentials
   ```

   Required variables:
   ```bash
   # ChromaDB Cloud (your specific instance)
   CHROMA_MODE=cloud
   CHROMA_API_KEY=your_api_key_here  # Get from ChromaDB Cloud dashboard
   CHROMA_TENANT=a41a948d-3ede-4ee4-a7be-efb41d8428d1
   CHROMA_DATABASE=CUDA-Weaver

   # Groq API
   GROQ_API_KEY=your_groq_api_key_here  # Get from console.groq.com
   ```

   **Getting ChromaDB Cloud API Key:**
   1. Go to your ChromaDB Cloud dashboard
   2. Navigate to your `CUDA-Weaver` database
   3. Click "Create API key" and copy it
   4. Paste it into your `.env` file as `CHROMA_API_KEY`

4. **Ingest knowledge base to ChromaDB Cloud**
   
   Upload CUDA documentation to your cloud database:
   ```bash
   python ingest/ingest_to_cloud.py
   ```
   
   This will upload documents from the `ingest/` directory to your ChromaDB Cloud instance.
   The script processes PDFs and HTML files and chunks them for optimal retrieval.

## Usage

### Interactive Mode (Recommended)

Run the interactive menu:
```bash
python run_interactive.py
```

The menu will guide you through:
1. Environment validation (ChromaDB, GPU detection)
2. GPU architecture selection
3. Search parameters (depth, candidates per round)
4. Log level configuration
5. Dry run option

Example session:
```
FSR CUDA Kernel Generator - Interactive Mode
============================================================

Environment Setup

ChromaDB Cloud configured: your-tenant/your-database
Auto-detected GPU: sm_86

GPU architecture [sm_86]: 

Search Configuration

Maximum search depth (iterations) [2]: 3
Candidates per iteration [2]: 5
Log level (DEBUG/INFO/WARNING) [INFO]: 

Dry run? (generate only, no compile/run) [y/N]: n

Configuration Summary:
============================================================
  GPU Architecture: sm_86
  Search Depth: 3
  Candidates/Round: 5
  Log Level: INFO
  Dry Run: False
  ChromaDB Mode: cloud
============================================================

Start FSR search? [Y/n]: y

Starting FSR search...
```

### Command-Line Mode (Legacy)

The original `run_fsr.py` still works:
```bash
python run_fsr.py --arch sm_86 --depth 3 --candidates 5
```

## Configuration Options

### ChromaDB Modes

**Cloud Mode (Default)** - Your Configuration
```bash
CHROMA_MODE=cloud
CHROMA_API_KEY=<get_from_chroma_dashboard>
CHROMA_TENANT=a41a948d-3ede-4ee4-a7be-efb41d8428d1
CHROMA_DATABASE=CUDA-Weaver
```

**Persistent Mode (Local)**
```bash
CHROMA_MODE=persistent
CHROMA_PERSIST_PATH=./chroma_db
```

### Search Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `depth` | Maximum FSR iterations | 2 |
| `candidates` | Kernels per iteration | 2 |
| `arch` | GPU architecture | auto-detect |

### Knowledge Base Stages

The RAG system retrieves from 5 hierarchical stages:
1. **Concepts** - CUDA threading, memory, synchronization
2. **Patterns** - Optimization patterns (tiling, unrolling)
3. **Hardware** - Architecture-specific limits (SM, registers)
4. **API** - CUDA API best practices
5. **Examples** - Reference implementations

## Project Structure

```
weaver-agent-workflow/
├── src/
│   ├── core/
│   │   ├── chroma_client.py       # ChromaDB client factory
│   │   ├── chroma_config.py       # Collection mappings
│   │   ├── docs_retriever.py      # Document retrieval
│   │   ├── vector_store.py        # Vector search utilities
│   │   ├── hierarchical_rag.py    # RAG orchestration
│   │   ├── fsr_framework.py       # Main FSR logic
│   │   ├── llm_interface.py       # LLM client (Groq)
│   │   ├── prompt_manager.py      # Prompt engineering
│   │   ├── benchmarks.py          # Task definitions
│   │   └── retrievers/            # Knowledge retrievers
│   ├── execution_checker/
│   │   ├── compilation_checker.py # nvcc compilation
│   │   ├── functional_validator.py # Correctness testing
│   │   └── performance_profiler.py # CUDA event timing
│   └── schemas/
│       ├── knowledge.py           # Knowledge schemas
│       └── llm_response.py        # LLM response schemas
├── ingest/
│   ├── ingest_to_cloud.py         # Cloud ingestion script
│   └── [concepts/patterns/etc]    # Documentation files
├── run_interactive.py             # Interactive menu runner
├── run_fsr.py                     # Legacy CLI runner
├── .env.template                  # Environment template
└── README.md                      # This file
```

## How It Works

### FSR Search Loop

```python
for iteration in range(max_depth):
    # 1. Generate candidates
    candidates = llm.generate_kernels(prompt, N)
    
    # 2. Compile
    compiled = [compile(c) for c in candidates]
    
    # 3. Validate correctness
    validated = [validate(c) for c in compiled if c.ok]
    
    # 4. Profile performance
    fastest = min(profile(c) for c in validated)
    
    # 5. Refine prompt
    if fastest:
        prompt = refine_for_performance(prompt, fastest)
    else:
        prompt = refine_for_errors(prompt, errors)
```

### Prompt Refinement

- **Initial**: Task + architecture + RAG knowledge
- **Error-based**: Add compilation/validation errors
- **Performance-based**: Add optimization hints + best kernel notes

## Example Output

```
FSR Search Results
============================================================
  Iterations: 3
  Total Candidates: 15
  Best Time: 0.042 ms
============================================================

Best kernel saved: best_kernel.cu
```

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
uv sync --refresh
```

### ChromaDB Connection Errors
```bash
# Verify credentials are set
echo $CHROMA_API_KEY  # Should show your API key
echo $CHROMA_TENANT   # Should show: a41a948d-3ede-4ee4-a7be-efb41d8428d1
echo $CHROMA_DATABASE # Should show: CUDA-Weaver

# Test connection to ChromaDB Cloud
python -c "from src.core.chroma_client import get_chroma_client; client = get_chroma_client(); print('Connected to ChromaDB Cloud')"
```

**If connection fails:**
- Verify your API key is correct in `.env`
- Check you have internet connectivity
- Ensure your ChromaDB Cloud instance is active
- Try regenerating your API key in the ChromaDB dashboard

### NVCC Not Found
```bash
# Set NVCC path
export NVCC=/usr/local/cuda/bin/nvcc
# or in .env
NVCC=/usr/local/cuda/bin/nvcc
```

### GPU Not Detected
```bash
# Check nvidia-smi
nvidia-smi

# Manually specify architecture
# In interactive mode, enter manually
# Or set in .env
GPU_ARCH=sm_80
```

## Development

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .
```

### Adding New Benchmarks

Edit `src/core/benchmarks.py`:
```python
def my_task() -> BenchmarkTask:
    desc = "Your task description for the LLM..."
    return BenchmarkTask("my_task", desc, "", [1<<20], ["tag1"])
```

### Extending RAG Knowledge

1. Add documents to `ingest/<stage>/`
2. Update `ingest/ingest_to_cloud.py` file list
3. Run ingestion: `python ingest/ingest_to_cloud.py`

## License

[Your License Here]

## Acknowledgments

- ChromaDB for vector storage
- LangChain for LLM orchestration
- Groq for fast LLM inference
- NVIDIA CUDA Toolkit