"""KitchenBot — a pip-installable pipeline for a tiny from-scratch cooking SLM.

The package is split by *job*, not by script order:

    config      paths + hyperparameters (single source of truth)
    data        corpus download, recipe formatting, synthetic Q&A
    tokenization BPE training, fast tokenizer, chat template
    training    pretrain + LoRA SFT
    modeling    model construction, loading, generation
    hub         model-card upload
    cli         argparse front end (`python -m kitchenbot ...`)

Every heavy dependency (torch, transformers, trl, ...) is imported lazily so
`python -m kitchenbot --help` works in a bare Python install.
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
