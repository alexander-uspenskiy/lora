# LoRA Fine-Tuning Project

This project demonstrates how to fine-tune a GPT-2 model using Low-Rank Adaptation (LoRA) on the Wikitext-2 dataset. The fine-tuned model is then used to generate text based on user-provided prompts.

## Prerequisites

- Python 3.7 or higher
- `pip` package manager

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/alexander-uspenskiy/lora
    cd lora
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv lora_env
    source lora_env/bin/activate  # On Windows, use `lora_env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Training

To fine-tune the model, run the [main.py] script:
```sh
python main.py
```

## External Link

https://dev.to/alexander_uspenskiy_the_great/who-is-lora-and-why-is-it-more-effective-then-just-training-of-your-llm-5793