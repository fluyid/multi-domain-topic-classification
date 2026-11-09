"""Pick the fasted available hardware for running PyTorch models."""

def get_device() -> str:
    """Return "cuda", "mps" or "cpu", whichever is the best one that's available.

    Order of precedence:
      1. "cuda": NVIDIA GPUs (Google Colab)
      2. "mps": Apple Silicon GPUs (the M2 MacBook)
      3. "cpu": works everywhere
    """
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed, so using: cpu")
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return device
