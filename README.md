# Tokenizer

This is the repository for the Tokenizer assignment of course CS-5-45(MO) [Natural Language Processing], 2026.

## Installation

We use `uv` for project and dependency management as well as package management.

> Install `uv` [here](https://docs.astral.sh/uv/getting-started/installation/)!
>
> OR just install using pip, `pip install uv`

Run the follwing commands to start the virtual environment:

```sh
uv sync
```

## Virtual Environment

To activate the environment, run:

```sh
./.venv/Scripts/activate # Windows
source .venv/bin/activate # Linux
```

To add any dependencies, use:

```sh
uv add <package-name>
# OR
uv add <package-name>==<pkg-version>

# example:
# uv add python-dotenv
# uv add python-dotenv==1.2.1
```

## Run

To run any file:

```sh
uv run <file-name>
```
