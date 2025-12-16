"""
Extract 32-frame video clips from BDD-X dataset

Reference: scripts/load_frames_sample.py and scripts/align_export_example.py

TSV format:
sample_id \t metadata_json \t frame1_b64 \t frame2_b64 \t ... \t frame32_b64
(34 columns total: 1 ID + 1 meta + 32 frames)
"""
import os
import json
import base64
from pathlib import Path
from io import BytesIO

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Missing dependencies:")
    print("   pip install Pillow numpy")
    exit(1)

# Relative paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSE595_ROOT = PROJECT_ROOT.parent

BDDX_DIR = CSE595_ROOT / "datasets/BDDX"
OUTPUT_DIR = SCRIPT_DIR / "extracted_samples"


class TSVFile:
    """
    Fast TSV file access using lineidx
    Reference: scripts/load_frames_sample.py
    """
    def __init__(self, tsv_path):
        self.tsv_path = Path(tsv_path)

        # Try to read .lineidx.8b (binary format, faster)
        # TSV path: xxx.img.tsv -> Index path: xxx.img.lineidx.8b
        base = str(self.tsv_path).rsplit('.', 1)[0]  # Remove .tsv
        li8_path = Path(base + '.lineidx.8b')
        li_path = Path(base + '.lineidx')

        if li8_path.exists():
            print(f"  Using index: {li8_path.name}")
            with open(li8_path, 'rb') as f:
                b = f.read()
            # Each 8 bytes is an offset (little-endian)
            self.offsets = [int.from_bytes(b[i:i+8], 'little') for i in range(0, len(b), 8)]
        elif li_path.exists():
            print(f"  Using index: {li_path.name}")
            with open(li_path, 'r') as f:
                self.offsets = [int(x.strip()) for x in f]
        else:
            raise FileNotFoundError(f"Index file not found: {li8_path} or {li_path}")

        self.fp = open(tsv_path, 'rb')
        print(f"  Loaded TSV: {self.tsv_path.name} ({len(self)} rows)")

    def __len__(self):
        return len(self.offsets)

    def get(self, i):
        """
        Get the contents of row i

        Returns:
            list of str: [sample_id, metadata_json, frame1_b64, ..., frame32_b64]
        """
        if i < 0 or i >= len(self.offsets):
            raise IndexError(f"Index {i} out of range [0, {len(self.offsets)-1}]")

        # Seek to the start position of this row
        self.fp.seek(self.offsets[i])
        # Read one line
        line = self.fp.readline().rstrip(b'\n')
        # Split by tab and decode to UTF-8
        return [p.decode('utf-8') for p in line.split(b'\t')]

    def __del__(self):
        if hasattr(self, 'fp'):
            self.fp.close()


def decode_b64_image(b64_str):
    """
    Decode a base64-encoded image

    Args:
        b64_str: base64 encoded string

    Returns:
        PIL Image or numpy array
    """
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return img


def extract_sample(tsv_file, caption_tsv, linelist_path, index, output_dir):
    """
    Extract all information for a sample

    Args:
        tsv_file: TSVFile object (image data)
        caption_tsv: TSVFile object (text data)
        linelist_path: linelist file path
        index: sample index
        output_dir: output directory
    """
    # Read linelist to get row index and caption index mapping
    with open(linelist_path, 'r') as f:
        pairs = [tuple(map(int, x.strip().split('\t'))) for x in f]

    if index < 0 or index >= len(pairs):
        raise IndexError(f'Index {index} out of range [0, {len(pairs)-1}]')

    row_idx, cap_idx = pairs[index]
    print(f"\nSample {index}: row_idx={row_idx}, cap_idx={cap_idx}")

    # 1. Read image data
    cols = tsv_file.get(row_idx)
    sample_id = cols[0]
    metadata = json.loads(cols[1])
    frame_b64_list = cols[2:]  # Remaining columns are base64-encoded frames

    print(f"  Sample ID: {sample_id}")
    print(f"  Metadata: {metadata}")
    print(f"  Number of frames: {len(frame_b64_list)}")

    # 2. Read caption data
    cap_cols = caption_tsv.get(row_idx)
    cap_sample_id = cap_cols[0]
    caps = json.loads(cap_cols[1])
    cap_data = caps[cap_idx]

    action = cap_data.get('action', '')
    justification = cap_data.get('justification', '')
    caption = cap_data.get('caption') or (action + ' ' + justification).strip()

    print(f"  Action: {action}")
    print(f"  Reason: {justification}")

    # 3. Create output directory
    sample_dir = output_dir / f"sample_{index:04d}_{sample_id.replace('/', '_')}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 4. Decode and save all frames
    print(f"  Decoding 32 frames...")
    frames = []
    for i, b64 in enumerate(frame_b64_list):
        img = decode_b64_image(b64)
        frames.append(img)

        # Save as JPG
        frame_path = sample_dir / f"frame_{i:02d}.jpg"
        img.save(frame_path, quality=95)

        if (i+1) % 8 == 0:
            print(f"    Saved {i+1}/{len(frame_b64_list)} frames...", end='\r')

    print(f"    Saved {len(frames)} frames                  ")

    # 5. Save metadata
    metadata_json = {
        'index': index,
        'sample_id': sample_id,
        'action': action,
        'justification': justification,
        'caption': caption,
        'metadata': metadata,
        'num_frames': len(frames),
        'frame_width': frames[0].width,
        'frame_height': frames[0].height,
    }

    json_path = sample_dir / "metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_json, f, indent=2, ensure_ascii=False)

    print(f"  Saved metadata: {json_path.name}")
    print(f"  Output directory: {sample_dir.name}/")

    return metadata_json


def main():
    print("="*70)
    print("BDD-X Sample Extractor")
    print("="*70)

    # Use training data
    split = "training"
    yaml_name = f"{split}_32frames.yaml"

    img_tsv_path = BDDX_DIR / "frame_tsv" / f"{split}_32frames_img_size256.img.tsv"
    caption_tsv_path = BDDX_DIR / f"{split}.caption.tsv"
    linelist_path = BDDX_DIR / f"{split}.caption.linelist.tsv"  # Use caption.linelist instead of linelist

    # Check file existence
    for path in [img_tsv_path, caption_tsv_path, linelist_path]:
        if not path.exists():
            print(f"File not found: {path}")
            return

    print(f"\nDataset: {split}")
    print(f"Image TSV: {img_tsv_path.name}")
    print(f"Caption TSV: {caption_tsv_path.name}")
    print(f"Linelist: {linelist_path.name}")

    # Create TSV readers
    print(f"\nLoading data...")
    img_tsv = TSVFile(img_tsv_path)
    cap_tsv = TSVFile(caption_tsv_path)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Extract first few samples
    n_samples = 3
    print(f"\n{'='*70}")
    print(f"Extracting first {n_samples} samples")
    print(f"{'='*70}")

    samples_info = []
    for i in range(n_samples):
        try:
            metadata = extract_sample(img_tsv, cap_tsv, linelist_path, i, OUTPUT_DIR)
            samples_info.append(metadata)
        except Exception as e:
            print(f"Failed to extract sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(samples_info, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Extracted {len(samples_info)} samples, 32 frames each")
    print(f"\nSummary file: {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
