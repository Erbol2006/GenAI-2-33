# GenAI-2-33
В данном репозитории находится решение для предмета "Введение в проектную деятельность". 2 блока задач - GenAI-2-33


# DeepSeek-R1 Advice Generator

Python script that utilizes the DeepSeek-R1 language model to generate personalized advice and recommendations in Russian.

## What it does

- Loads the DeepSeek-R1 8B model from Unsloth
- Processes Russian-language requests for advice
- Generates numbered recommendations on various topics
- Supports both basic advice and prioritized recommendations
- Validates output formatting and numbering

## Key Features

- **Dual Generation Modes**: Basic advice and importance-prioritized suggestions
- **Russian Language Support**: Optimized for Russian prompts and responses
- **Input Validation**: Parses and validates user requests
- **Numbering Verification**: Ensures proper sequential numbering in responses
- **Performance Testing**: Includes comprehensive test functions


## bash
pip install transformers accelerate torch unsloth


## Usage

### Basic advice generation:

advice = generate_advice("Дай 3 совета по учебе")
print(advice)


### Prioritized advice generation:

prioritized_advice = generate_prioritized_advice("Дай 4 совета по программированию, пронумеруй по важности")
print(prioritized_advice)


### Run tests:

test_basic_advice()
test_prioritized_advice()


## Model Information

- Model: DeepSeek-R1-Qwen3-8B
- Architecture: Qwen-based causal language model
- Context Window: 2048 tokens
- Precision: Float16 for efficiency

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for optimal performance

