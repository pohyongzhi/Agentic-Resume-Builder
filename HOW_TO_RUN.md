# How to run the V5 workflow experiment

## First-time setup

```bash
git clone https://github.com/pohyongzhi/Agentic-Resume-Builder.git
cd Agentic-Resume-Builder
git switch v5-workflow

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3.5:4b-q4_K_M
```

Install the public fine-tuned GGUF without a Hugging Face account:

```bash
MODEL_DIR="$HOME/Agentic-Resume-Builder-Models/fine-tuned-v1"
mkdir -p "$MODEL_DIR"

curl -L -C - \
  -o "$MODEL_DIR/agentic-resume-v4-1-no-nextn-q4_K_M.gguf" \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/agentic-resume-v4-1-no-nextn-q4_K_M.gguf

curl -L \
  -o "$MODEL_DIR/Modelfile" \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/Modelfile

ollama create agentic-resume-v4-1:q4_K_M -f "$MODEL_DIR/Modelfile"
```

Confirm the tags:

```bash
ollama show qwen3.5:4b-q4_K_M
ollama show agentic-resume-v4-1:q4_K_M
```

## Web application

```bash
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:7860>. The selector shows only implemented variants:

- Variant 1 — One-pass
- Variant 5 — Fine-tuned small generator

## Evaluation CLI

```bash
python run_all.py --workflows one_pass --limit 2 --repeats 1
python run_all.py --workflows fine_tuned_small_generator --limit 2 --repeats 1
python run_all.py --workflows one_pass fine_tuned_small_generator --limit 2 --repeats 1
```

Variants 2, 3, 4, 6 and 7 are registered as planned and cannot be executed until their documented techniques are implemented.

If Ollama refuses the connection, start the Ollama application or `ollama serve`. If a tag is missing, inspect `ollama list`. Results include the actual model tag in every LLM trace.
