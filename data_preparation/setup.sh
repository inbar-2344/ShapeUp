#!/bin/bash
# Install the system libraries that headless Blender (the `bpy` Python module)
# needs to import and render off-screen. Run once on a fresh machine.
#
#   sudo bash setup.sh      # (or run as root, e.g. inside a container)
set -euo pipefail

apt-get update
apt-get install -y \
    libxrender1 \
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxkbcommon0 \
    libgl1 \
    libsm6

echo "System dependencies installed. Next:"
echo "  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121"
