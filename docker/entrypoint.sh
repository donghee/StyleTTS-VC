#!/bin/bash
set -e
eval "$(pyenv init -)"
pip install -r requirements.txt
exec "$@"
