"""
Simple example: Get BDD-X triplet (video, action, reason)

Uses relative paths for portability across servers
"""
import json
from pathlib import Path

# Relative path configuration
SCRIPT_DIR = Path(__file__).parent  # examples/
PROJECT_ROOT = SCRIPT_DIR.parent    # driving-explanation-retrieval/
CSE595_ROOT = PROJECT_ROOT.parent   # cse595/

# Data path
BDDX_JSON = CSE595_ROOT / "datasets/BDDX/captions_BDDX.json"


def get_triplet_text(index=0):
    """
    Get text data for a sample

    Args:
        index: sample index

    Returns:
        sample_id: sample ID (corresponds to video)
        d: action description
        e: explanation/reason
    """
    # Check if file exists
    if not BDDX_JSON.exists():
        print(f"Error: Data file not found {BDDX_JSON}")
        print(f"Please verify directory structure:")
        print(f"  cse595/")
        print(f"  ├── driving-explanation-retrieval/")
        print(f"  └── datasets/BDDX/captions_BDDX.json")
        return None, None, None

    # Load data
    with open(BDDX_JSON, 'r') as f:
        data = json.load(f)

    # Get sample
    sample = data['annotations'][index]

    sample_id = sample['vidName']
    d = sample['action']
    e = sample['justification']

    return sample_id, d, e


def show_multiple_examples(n=5):
    """Display multiple samples"""
    if not BDDX_JSON.exists():
        print(f"Error: Data file not found {BDDX_JSON}")
        return

    with open(BDDX_JSON, 'r') as f:
        data = json.load(f)

    print("="*70)
    print(f"Displaying first {n} BDD-X samples")
    print("="*70)

    for i in range(n):
        sample = data['annotations'][i]
        print(f"\nSample {i+1}:")
        print(f"  ID:     {sample['vidName']}")
        print(f"  Action: {sample['action']}")
        print(f"  Reason: {sample['justification']}")
        print(f"  Time:   {sample['sTime']}s - {sample['eTime']}s")


if __name__ == "__main__":
    print("BDD-X Triplet Example")
    print("="*70)
    print(f"Current directory: {Path.cwd()}")
    print(f"Data file path: {BDDX_JSON}")
    print(f"Data file exists: {BDDX_JSON.exists()}")

    # Example 1: Get single triplet
    sample_id, d, e = get_triplet_text(index=0)

    if sample_id is None:
        print("\nFailed to load data")
        exit(1)

    print("\nSingle sample:")
    print(f"  Sample ID (v): {sample_id}")
    print(f"  Action (d):    {d}")
    print(f"  Reason (e):    {e}")

    # Example 2: Display multiple samples
    print("\n")
    show_multiple_examples(n=5)

    print("\n" + "="*70)
    print("Notes:")
    print("  - v (video): video frames for sample_id")
    print("  - d (action): action description (query)")
    print("  - e (explanation): explanation/reason (target)")
    print("="*70)
