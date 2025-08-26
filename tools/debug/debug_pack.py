#!/usr/bin/env python3
"""Debug pack structure."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from packrepo.library import RepositoryPacker

packer = RepositoryPacker()
pack = packer.pack_repository(
    project_root,
    token_budget=5000,
    variant="baseline",
    deterministic=True,
    enable_oracles=False
)

print("Pack index attributes:")
for attr in dir(pack.index):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(pack.index, attr, 'N/A')}")
        
print(f"\nPack type: {type(pack)}")
print(f"Index type: {type(pack.index)}")