# picoLLM

This project is about building and studying a **very small language model** (a “pico” LLM).  
The main goal is to understand how language models work by training them on simple, small datasets instead of huge internet text.

---

## What this project does

- Trains a tiny language model on short text sequences.
- Lets you change settings (like model size or training steps) and see what happens.
- Saves results so you can compare different runs.
- Includes write-ups that explain the experiments and what we learned.

This is a learning project for class and for building my skills in machine learning.

---

## Main files and folders

- `pico-llm.py` – main Python script for training and testing the model.  
- `pico-llm.ipynb` – Jupyter notebook version to run and see results step by step.  
- `data/` – training data (small text or token sequences).  
- `artifacts/` – saved models, logs, and plots from training runs.  
- `monitor_and_plot.py` and other `collect_*.py` / `.sh` – helper scripts for running many experiments and plotting results.  
- `pico_llm_action_plan.*` and `tasks.*` – project reports and homework write-ups (LaTeX and PDF).

---

## How to run the code

1. **(Optional) Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# or
.\.venv\Scripts\activate    # Windows
```

2. **Install Python packages**

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If not, install some basic packages:

```bash
pip install torch numpy matplotlib
```

3. **Run a basic training run**

```bash
python pico-llm.py
```

4. **See all command-line options**

```bash
python pico-llm.py --help
```



