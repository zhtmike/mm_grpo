# Installation

mm_grpo is a Python library that mainly contains the python implementations for multimodal reinforcement learning frameworks and models. This guide will help you install MM-GRPO and all its dependencies.

## Prerequisites

- OS: Linux
- Python 3.10 or higher
- CUDA-capable GPU

## Installation

### Install from Source

The latest version of `gerl` can be installed as follows:

```bash
pip install git+https://github.com/leibniz-csi/mm_grpo.git
```

### Install from Local

If you prefer to use the scripts under `examples/` directly, please clone the repository and install the package locally:

```bash
git clone https://github.com/leibniz-csi/mm_grpo.git && cd mm_grpo
pip install -e . 

# with paddleocr reward support
# pip install -e .[paddleocr]
```