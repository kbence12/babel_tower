"""Fine-tuning a pretrained translation model (OPUS-MT / Marian) on the book corpus.

Written as a plain PyTorch loop rather than with Hugging Face's Trainer, so that every step
(batching, loss, optimiser, learning-rate schedule, evaluation) is visible and can be changed.

  data.py   corpus pairs -> token ids -> padded batches
"""
