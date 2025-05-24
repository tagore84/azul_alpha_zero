#!/usr/bin/env bash
set -e

# 1) crea/activa tu virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) actualiza pip e instala deps
pip install --upgrade pip
pip install -r requirements.txt