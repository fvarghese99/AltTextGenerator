# GPU Installs

## Apple Silicon (MPS):

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio # Mac MPS version
```

## Windows (NVIDIA GPU):

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## CPU-only (fallback on either platform):

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
