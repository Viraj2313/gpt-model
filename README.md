# My Own GPT Model

I have always been fascinated by how GPT models work, how they generate text, how attention operates, and how the underlying architecture comes together. One day, I decided to build my own GPT-style language model from scratch. That began a long cycle of reading research papers, studying the _Attention Is All You Need_ paper, understanding the Transformer architecture, and strengthening my fundamentals.

This project is the result of that journey. I implemented a GPT-like model with roughly **60 million parameters**, trained it using **PyTorch** on Kaggle GPUs over several days. The model was trained on **WikiText** and **DailyDialog** datasets.

---

## Repository Structure

```
gpt-model/
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── data/
│   │   ├── bpe.py
│   │   ├── prepare_data.py
│   │   └── __init__.py
│   │
│   ├── infer/
│   │   ├── generate.py
│   │   └── __init__.py
│   │
│   ├── model/
│   │   ├── gpt.py
│   │   └── __init__.py
│   │
│   ├── training/
│   │   ├── load.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── chat.py
├── train.py
├── requirements.txt
└── README.md
```

---

## How to Use This Repository

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

```bash
python src/data/prepare_data.py
```

This generates tokenized and processed data required for training.

### 3. Train the Model

```bash
python train.py
```

This script initializes the GPT model, loads the processed data, and begins training. Checkpoints are saved automatically.

### 4. Generate Text

```bash
python chat.py
```

or

```bash
python src/infer/generate.py
```

Both scripts load the trained checkpoint and generate text interactively.

### 5. Using the Notebook

The full experiment pipeline, including training logic, is available in:

```
notebooks/training.ipynb
```

---

## Highlights

- Custom Byte Pair Encoding (BPE) tokenizer
- GPT-style Transformer architecture built from scratch using PyTorch
- Modular project structure for research and extension
- Training and inference scripts included
- Fully reproducible pipeline
