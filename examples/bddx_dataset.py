"""
BDD-X Dataset for Driving Explanation Retrieval

Provides PyTorch Dataset class for convenient training
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset


class BDDXRetrievalDataset(Dataset):
    """
    BDD-X Driving Explanation Retrieval Dataset

    For training triplet retrieval models
    Input: (video, action description)
    Target: explanation

    Usage:
        # Create dataset
        dataset = BDDXRetrievalDataset(split='training')

        # Get single sample
        item = dataset[0]
        # item = {
        #     'sample_id': '06d501fd-a9ffc960',
        #     'action': 'The car accelerates',
        #     'explanation': 'because the light has turned green.',
        #     'start_time': 0,
        #     'end_time': 11
        # }

        # Use with DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch in loader:
            actions = batch['action']
            explanations = batch['explanation']
            # ... training code
    """

    def __init__(
        self,
        split: str = 'training',
        data_root: Optional[Path] = None,
        load_all: bool = True
    ):
        """
        Args:
            split: 'training', 'validation', or 'testing'
            data_root: data root directory (optional, auto-detect by default)
            load_all: whether to load all data at once (recommended True for speed)
        """
        super().__init__()

        self.split = split
        self.load_all = load_all

        # Auto-detect data path
        if data_root is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            cse595_root = project_root.parent
            data_root = cse595_root / "datasets/BDDX"

        self.data_root = Path(data_root)
        self.json_path = self.data_root / "captions_BDDX.json"

        # Check if file exists
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.json_path}\n"
                f"Please verify directory structure"
            )

        # Load data
        self._load_data()

        print(f"Loaded {split} dataset: {len(self)} samples")

    def _load_data(self):
        """Load JSON data and filter by split"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        all_annotations = data['annotations']

        # Filter data by split
        # BDD-X naming convention: training_xxx, validation_xxx, testing_xxx
        # But vidName in JSON has no prefix, need to split by index

        # According to official statistics:
        # training: 21,143
        # validation: 2,519
        # testing: 2,859
        # total: 26,521

        if self.split == 'training':
            self.annotations = all_annotations[:21143]
        elif self.split == 'validation':
            self.annotations = all_annotations[21143:21143+2519]
        elif self.split == 'testing':
            self.annotations = all_annotations[21143+2519:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'training', 'validation', or 'testing'")

        # Build explanation bank (all unique explanations)
        self.explanation_bank = self._build_explanation_bank()

    def _build_explanation_bank(self) -> List[str]:
        """
        Build explanation candidate set for current split

        Returns:
            List[str]: all unique explanations
        """
        explanations = set()
        for item in self.annotations:
            exp = item['justification'].strip().lower()
            explanations.add(exp)

        # Convert to list and sort (for reproducibility)
        exp_list = sorted(list(explanations))
        return exp_list

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get single sample

        Args:
            idx: sample index

        Returns:
            dict: {
                'sample_id': str,
                'action': str,
                'explanation': str,
                'start_time': int,
                'end_time': int,
                'index': int  # original index
            }
        """
        sample = self.annotations[idx]

        return {
            'sample_id': sample['vidName'],
            'action': sample['action'],
            'explanation': sample['justification'],
            'start_time': int(sample['sTime']),
            'end_time': int(sample['eTime']),
            'index': idx
        }

    def get_explanation_bank(self) -> List[str]:
        """
        Get all candidate explanations for current split

        Used for retrieval task: retrieve correct explanation from this bank

        Returns:
            List[str]: all unique explanations
        """
        return self.explanation_bank

    def get_negative_samples(
        self,
        idx: int,
        n: int = 9,
        random_state: int = 42
    ) -> List[str]:
        """
        Get negative sample explanations for a given sample

        Used for closed-set retrieval (e.g., N=10, 1 positive + 9 negatives)

        Args:
            idx: sample index
            n: number of negative samples (default 9)
            random_state: random seed

        Returns:
            List[str]: n negative sample explanations
        """
        import random

        # Get positive sample's explanation
        gold_exp = self.annotations[idx]['justification'].strip().lower()

        # Exclude positive sample from bank
        candidates = [e for e in self.explanation_bank if e != gold_exp]

        # Random sampling
        random.seed(random_state + idx)  # for reproducibility
        negatives = random.sample(candidates, min(n, len(candidates)))

        return negatives

    def get_stats(self) -> Dict:
        """
        Get dataset statistics

        Returns:
            dict: statistics
        """
        actions = [s['action'] for s in self.annotations]
        explanations = [s['justification'] for s in self.annotations]

        return {
            'split': self.split,
            'num_samples': len(self),
            'num_unique_explanations': len(self.explanation_bank),
            'avg_action_length': sum(len(a.split()) for a in actions) / len(actions),
            'avg_explanation_length': sum(len(e.split()) for e in explanations) / len(explanations),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader

    Organizes batch samples into dictionary format

    Args:
        batch: List of dicts from __getitem__

    Returns:
        dict: Batched data
    """
    return {
        'sample_ids': [item['sample_id'] for item in batch],
        'actions': [item['action'] for item in batch],
        'explanations': [item['explanation'] for item in batch],
        'start_times': torch.tensor([item['start_time'] for item in batch]),
        'end_times': torch.tensor([item['end_time'] for item in batch]),
        'indices': torch.tensor([item['index'] for item in batch])
    }


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_usage():
    """Example 1: Basic usage"""
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)

    # Create dataset
    dataset = BDDXRetrievalDataset(split='training')

    # View statistics
    stats = dataset.get_stats()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get single sample
    print(f"\nFirst sample:")
    item = dataset[0]
    for key, value in item.items():
        print(f"  {key}: {value}")

    # Get negative samples
    print(f"\nGenerating 9 negative samples for first sample:")
    negatives = dataset.get_negative_samples(idx=0, n=9)
    for i, neg in enumerate(negatives[:3], 1):
        print(f"  {i}. {neg}")
    print(f"  ... ({len(negatives)} total)")


def example_dataloader():
    """Example 2: Using with PyTorch DataLoader"""
    print("\n" + "="*70)
    print("Example 2: Using with DataLoader")
    print("="*70)

    from torch.utils.data import DataLoader

    # Create dataset
    dataset = BDDXRetrievalDataset(split='training')

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Iterate one batch
    print(f"\nGetting one batch (batch_size=4):")
    batch = next(iter(loader))

    print(f"\nBatch keys: {batch.keys()}")
    print(f"\nBatch sample_ids: {batch['sample_ids']}")
    print(f"\nBatch actions (first 2):")
    for i, action in enumerate(batch['actions'][:2]):
        print(f"  {i+1}. {action}")
    print(f"\nBatch explanations (first 2):")
    for i, exp in enumerate(batch['explanations'][:2]):
        print(f"  {i+1}. {exp}")


def example_retrieval_setup():
    """Example 3: Retrieval task setup"""
    print("\n" + "="*70)
    print("Example 3: Retrieval Task Setup (Closed-set N=10)")
    print("="*70)

    dataset = BDDXRetrievalDataset(split='validation')

    # Build retrieval candidate set for first sample
    idx = 0
    item = dataset[idx]

    print(f"\nSample {idx}:")
    print(f"  Action: {item['action']}")
    print(f"  Gold explanation: {item['explanation']}")

    # Get 9 negative samples
    negatives = dataset.get_negative_samples(idx=idx, n=9)

    # Build candidate set (1 gold + 9 negatives)
    candidates = [item['explanation']] + negatives

    print(f"\nCandidate set (N=10):")
    for i, cand in enumerate(candidates, 1):
        prefix = "GOLD" if i == 1 else "  "
        print(f"  {i}. {prefix} {cand}")

    print(f"\nTask: Given action and 10 candidate explanations, select the correct one")


if __name__ == "__main__":
    print("BDD-X Retrieval Dataset Examples")
    print("="*70)

    # Run all examples
    example_basic_usage()
    example_dataloader()
    example_retrieval_setup()

    print("\n" + "="*70)
    print("All examples completed!")
    print("\nRecommendations:")
    print("  1. Use DataLoader + collate_fn for training")
    print("  2. Use get_negative_samples() for closed-set candidates")
    print("  3. Use get_explanation_bank() for open-set candidate set")
    print("="*70)
