"""Tests run on the CPU. Flash-attention would be selected as soon as a card
and the flash_attn package are both present, and its kernels do not exist for
the CPU backend, so the tests pin the library default."""

import os

os.environ.setdefault("GAVEL_ATTN", "default")
