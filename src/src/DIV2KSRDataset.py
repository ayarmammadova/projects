import json
from pathlib import Path
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset


class DIV2KSRDataset(Dataset):
    """
    Dataset for loading preprocessed DIV2K super-resolution data from sharded tensor files.
    
    Data format:
    - Sharded tensor files: lr_{split}_XXXX.pt, hr_{split}_XXXX.pt
    - Manifest file: manifest_{split}.json
    - Tensor format: uint8 (0-255), RGB, CxHxW
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: Directory containing the sharded tensor files and manifest
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to apply to the data
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load manifest
        manifest_path = self.root_dir / f"manifest_{split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}. "
                f"Make sure the data has been preprocessed."
            )
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.count = self.manifest['count']
        self.num_shards = self.manifest['num_shards']
        
        # Load all shards into memory (or implement lazy loading if memory is limited)
        self.lr_shards = []
        self.hr_shards = []
        self.shard_sizes = []  # Number of items in each shard
        self.shard_cumsum = [0]  # Cumulative sum for indexing
        
        for shard_idx in range(self.num_shards):
            lr_path = self.root_dir / f"lr_{split}_{shard_idx:04d}.pt"
            hr_path = self.root_dir / f"hr_{split}_{shard_idx:04d}.pt"
            
            if not lr_path.exists() or not hr_path.exists():
                raise FileNotFoundError(
                    f"Shard file not found: {lr_path} or {hr_path}"
                )
            
            # Load shards
            lr_shard = torch.load(lr_path, map_location='cpu')
            hr_shard = torch.load(hr_path, map_location='cpu')
            
            # Verify shapes match
            assert lr_shard.shape[0] == hr_shard.shape[0], \
                f"Mismatch in shard {shard_idx}: LR has {lr_shard.shape[0]} items, HR has {hr_shard.shape[0]}"
            
            self.lr_shards.append(lr_shard)
            self.hr_shards.append(hr_shard)
            self.shard_sizes.append(lr_shard.shape[0])
            self.shard_cumsum.append(self.shard_cumsum[-1] + lr_shard.shape[0])
        
        # Verify total count
        total_loaded = sum(self.shard_sizes)
        if total_loaded != self.count:
            print(f"Warning: Manifest says {self.count} items, but loaded {total_loaded} items")
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Index {idx} out of range [0, {self.count})")
        
        # Find which shard contains this index
        shard_idx = 0
        for i in range(len(self.shard_cumsum) - 1):
            if self.shard_cumsum[i] <= idx < self.shard_cumsum[i + 1]:
                shard_idx = i
                break
        
        # Get local index within the shard
        local_idx = idx - self.shard_cumsum[shard_idx]
        
        # Get the data
        lr = self.lr_shards[shard_idx][local_idx]  # Shape: (C, H, W)
        hr = self.hr_shards[shard_idx][local_idx]  # Shape: (C, H, W)
        
        # Convert from uint8 (0-255) to float32 (0.0-1.0)
        lr = lr.float() / 255.0
        hr = hr.float() / 255.0
        
        # Apply transform if provided
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        
        return lr, hr

