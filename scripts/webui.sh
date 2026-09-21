#!/usr/bin/env bash
# The working surface on http://127.0.0.1:8030 (and the Tailscale address).
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
[ -f ~/.config/gavel/keys.env ] && source ~/.config/gavel/keys.env
export MEMGUARD_ALLOW_MB=${MEMGUARD_ALLOW_MB:-16000} HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=src
exec .venv/bin/python -m gavel.webui "$@"
