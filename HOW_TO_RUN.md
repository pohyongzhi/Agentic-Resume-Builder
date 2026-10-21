# How to Run - Agentic Resume Builder

This project uses local LLMs via Ollama to tailor resumes to job descriptions. It includes a baseline approach and multiple agentic workflows.

## Prerequisites

### 1. Install Ollama
Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull Required Models
```bash
# Writer model (used by all workflows)
ollama pull qwen3.5:4b-q4_K_M

# Embedding model (for similarity calculations)
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Web Interface (Recommended for Quick Testing)
Launch the Gradio web interface:
```bash
python app.py
```

Then open your browser to: **http://127.0.0.1:7860**

**Usage:**
1. Upload your resume (.pdf, .docx, .txt, or .md)
2. Paste the job description
3. Optionally add comma-separated keywords to track (e.g., "SQL, Power BI, Python")
4. Click "Tailor my resume"
5. Download the tailored resume as Markdown

### Option 2: Command Line (Baseline)
Tailor a resume directly from the terminal:
```bash
python baseline_tailor.py --resume my_resume.pdf --jd job.txt
```

With custom output file:
```bash
python baseline_tailor.py --resume my_resume.docx --jd job.txt --out tailored.md
```

## Running Evaluations

### Full Evaluation Suite
Run all workflows on all test pairs with 2 repeats each:
```bash
python run_all.py
```

### Quick Test
Run on first 2 pairs with 1 repeat:
```bash
python run_all.py --limit 2 --repeats 1
```

### Run Specific Workflow
```bash
python run_all.py --workflows baseline
```

### Skip Faithfulness Check (Faster)
```bash
python run_all.py --skip-faithfulness
```

**Note:** Results are saved to `results/results.jsonl`. The script is safe to stop and restart - completed runs are automatically skipped.

## Configuration

Edit `config.py` to change:
- **MODEL**: The writer model (default: qwen3.5:4b-q4_K_M)
- **EMBED_MODEL**: Embedding model (default: nomic-embed-text)
- **TEMPERATURE**: Generation randomness (default: 0.0 for consistency)
- **NUM_CTX**: Context window size (default: 16384)
- **MAX_STEPS**: Agent step budget (default: 8)

## Project Structure

```
.
├── app.py                      # Gradio web interface
├── baseline_tailor.py          # Simple CLI tailor (no agents)
├── run_all.py                  # Evaluation runner
├── config.py                   # Shared configuration
├── requirements.txt            # Python dependencies
├── common/                     # Shared utilities (LLM, data, prompts)
├── workflows/                  # Different agentic approaches (W1-W5)
├── eval/                       # Metrics and evaluation code
├── data/                       # Test pairs (resume + JD + keywords)
└── results/                    # Evaluation outputs
```

## Troubleshooting

### "Model not found" error
Make sure you've pulled the models:
```bash
ollama pull qwen3.5:4b-q4_K_M
ollama pull nomic-embed-text
```

### "Connection refused" error
Ensure Ollama is running:
```bash
ollama serve
```

### PDF/DOCX reading issues
If you get errors reading files, make sure pypdf and python-docx are installed:
```bash
pip install pypdf python-docx
```

### First run is slow
The first model call takes longer as Ollama loads the model into memory. Subsequent calls are much faster.

## Next Steps

- Compare different workflows by running `python run_all.py`
- Analyze results with the summarization scripts in the project
- Experiment with different models by editing `config.py`
- Test with your own resume and job descriptions using the web interface
