# Research Paper Assistant: Custom GPT-2 Implementation

An end-to-end implementation of the GPT-2 transformer architecture from scratch using PyTorch. This project loads pre-trained HuggingFace weights and fine-tunes the model to act as an interactive, instruction-following research paper assistant.

## Key Features
* **Custom Architecture:** Complete implementation of Multi-Head Causal Self-Attention, Feed-Forward Networks, and Transformer Blocks from scratch.
* **Pre-trained Weight Transfer:** Seamlessly maps and loads `gpt2-small` weights from the HuggingFace Hub into the custom PyTorch model.
* **Supervised Fine-Tuning (SFT):** Instruction-tunes the model on dynamically generated QA pairs extracted from PDF research papers.
* **Interactive CLI Assistant:** A command-line interface that allows users to ingest PDFs, generate summaries, and ask context-specific questions.
* **Ollama Evaluation:** Integrates local LLM evaluation (via Ollama) to autonomously score the fine-tuned model's responses.

## Tech Stack
* **Deep Learning:** PyTorch, `torch.nn`
* **NLP:** `tiktoken` (tokenization), HuggingFace `transformers`
* **Document Processing:** PyMuPDF (`fitz`)
* **Evaluation:** Ollama API

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/piyushjat/research-paper-assistant.git
   cd research-paper-assistant

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. (Optional) Install and start Ollama locally for automated response evaluation.For this the code is given in the Project.ipynb file before stage2 code cell


## Project Pipeline

1. **Stage 1**: Tiktoken tokenization, sliding window data loading, and core transformer modeling.
2. **Stage 2**: AdamW optimization, cosine learning rate scheduling, cross-entropy + perplexity tracking.
3. **Stage 3**: Chunking PDF text with overlap, formatting instruction datasets, and generating autoregressive responses with temperature and top-k sampling.

**Folder Structure**
research-paper-assistant/  
│── requirements.txt  
│── gpt_architecture.ipynb  
│── train_and_evaluate.ipynb  
│── Project.ipynb  
│── README.md

