"""
Semantic clustering of BDD-X reasons/justifications using S-BERT

Pipeline:
1. Load all reasons
2. Text normalization (synonym merging)
3. Generate embeddings using text encoder (S-BERT / E5) with optional PCA reduction
4. K-Means clustering
5. Semantic post-processing: merge highly similar or small clusters
6. Output statistics, visualization, and persist results

Dependencies:
    pip install sentence-transformers scikit-learn matplotlib seaborn
"""
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None

# Fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = AutoModel = None

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score
    )
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("scikit-learn is required:")
    print("   pip install scikit-learn")
    raise

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    print("matplotlib/seaborn not installed, skipping visualization")
    PLOT_AVAILABLE = False


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSE595_ROOT = PROJECT_ROOT.parent
BDDX_JSON = CSE595_ROOT / "datasets/BDDX/captions_BDDX.json"
OUTPUT_DIR = SCRIPT_DIR / "clustering_results"


# Normalization rules: unify common synonyms/phrases to improve clustering stability
NORMALIZATION_RULES = [
    (r"\btraffic lights?\b", "traffic_light"),
    (r"\btraffic signals?\b", "traffic_light"),
    (r"\bstop lights?\b", "traffic_light"),
    (r"\blight at the intersection\b", "traffic_light"),
    (r"\bred lights?\b", "traffic_light_red"),
    (r"\bgreen lights?\b", "traffic_light_green"),
    (r"\byellow lights?\b", "traffic_light_yellow"),
    (r"\bpeds?\b", "pedestrian"),
    (r"\bpeople crossing\b", "pedestrian"),
    (r"\bbicyclist(s)?\b", "bicycle"),
    (r"\bbike(s)?\b", "bicycle"),
    (r"\bcyclist(s)?\b", "bicycle"),
    (r"\bcar(s)? ahead\b", "lead_vehicle"),
    (r"\bvehicle(s)? ahead\b", "lead_vehicle"),
    (r"\bcar(s)? in front\b", "lead_vehicle"),
    (r"\bslow traffic\b", "traffic_heavy"),
    (r"\bheavy traffic\b", "traffic_heavy"),
    (r"\bcongestion\b", "traffic_heavy"),
    (r"\bmerge(s|d|ing)?\b", "merge"),
    (r"\bturn(s|ed|ing)? left\b", "turn_left"),
    (r"\bturn(s|ed|ing)? right\b", "turn_right"),
    (r"\bconstruction\b", "road_construction"),
    (r"\broad work\b", "road_construction"),
    (r"\bwork zone\b", "road_construction"),
    (r"\broad debris\b", "road_debris"),
    (r"\bpothole\b", "pothole"),
    (r"\bspeed bump\b", "speed_bump"),
    (r"\bambulance\b", "emergency_vehicle"),
    (r"\bfire truck\b", "emergency_vehicle"),
    (r"\bpolice car\b", "emergency_vehicle"),
    (r"\btaxi\b", "taxi"),
    (r"\bbus\b", "bus"),
    (r"\brain(y)?\b", "rain"),
    (r"\bsnow(y)?\b", "snow"),
    (r"\bfog(gy)?\b", "fog"),
    (r"\bslippery\b", "slippery"),
    (r"\bclear road\b", "traffic_clear"),
    (r"\bno traffic\b", "traffic_clear"),
    (r"\bopen road\b", "traffic_clear"),
    (r"\bbecause\b", ""),
    (r"\bsince\b", ""),
    (r"\bdue to\b", ""),
    (r"\bas\b", "")
]

# Semantic keyword groups: clusters matching same group keywords will be force-merged
FORCED_KEYWORD_GROUPS = {
    "traffic_light_red": ["traffic_light_red", "traffic light red", "red_light", "red light"],
    "traffic_light_green": ["traffic_light_green", "traffic light green", "green_light", "green light"],
    "traffic_light_yellow": ["traffic_light_yellow", "traffic light yellow", "yellow_light", "yellow light"],
    "pedestrian": ["pedestrian", "people crossing"],
    "merge": ["merge"],
    "road_construction": ["road_construction", "road construction", "work zone"],
    "road_debris": ["road_debris", "road debris", "pothole", "speed bump"],
    "lead_vehicle": ["lead_vehicle", "lead vehicle", "car ahead", "vehicle ahead"],
}


def normalize_reason_text(text: str) -> str:
    """Unify text expressions to reduce noise"""
    normalized = text.lower()
    for pattern, replacement in NORMALIZATION_RULES:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9_ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_all_reasons(split: str = "all"):
    """Load all BDD-X reasons/actions"""
    with open(BDDX_JSON, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    if split == "training":
        annotations = annotations[:21143]
        split_name = "training"
    elif split == "validation":
        annotations = annotations[21143:21143 + 2519]
        split_name = "validation"
    elif split == "testing":
        annotations = annotations[21143 + 2519:]
        split_name = "testing"
    else:
        split_name = "all"

    reasons = []
    for i, sample in enumerate(annotations):
        global_idx = sample.get("id", i)
        if split_name == "all":
            if global_idx < 21143:
                sample_split = "training"
            elif global_idx < 21143 + 2519:
                sample_split = "validation"
            else:
                sample_split = "testing"
        else:
            sample_split = split_name

        reasons.append({
            "index": i,
            "global_index": sample["id"],
            "reason": sample["justification"].strip(),
            "action": sample["action"].strip(),
            "video_id": sample.get("vidName", sample.get("video_id", "")),
            "split": sample_split
        })

    print(f"Loaded {len(reasons)} reasons (split={split_name})")
    return reasons


def encode_texts(texts, model_name, backend="sbert", batch_size=64):
    backend = backend.lower()

    if backend == "sbert":
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed, cannot use S-BERT")
        model = SentenceTransformer(model_name)
        return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    if backend == "e5":
        if AutoTokenizer is None or AutoModel is None or torch is None:
            raise ImportError("E5 requires transformers and torch to be installed")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # E5 recommends prepending "query:" to inputs
                batch_inputs = [f"query: {x}" for x in batch]
                inputs = tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = torch.sum(hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts
                embeddings.extend(mean_pooled.cpu().numpy())
        return np.array(embeddings)

    raise ValueError(f"Unsupported backend: {backend}")


def cluster_with_embeddings(
    reasons_data,
    model_name: str = "all-MiniLM-L6-v2",
    backend: str = "sbert",
    n_clusters: int = 35,
    random_state: int = RANDOM_SEED,
    normalize_text: bool = True,
    use_pca: bool = True,
    pca_components: int = 128,
    precomputed_embeddings: np.ndarray = None
):
    """Cluster using specified text encoder + (optional PCA) + KMeans"""
    print("\n" + "=" * 70)
    print(f"Clustering with text encoder ({backend})")
    print("=" * 70)

    print(f"\nLoading model: {model_name}")

    reasons_text = []
    for item in reasons_data:
        if normalize_text:
            normalized = normalize_reason_text(item["reason"])
            item["normalized_reason"] = normalized
            reasons_text.append(normalized)
        else:
            item["normalized_reason"] = item["reason"].lower()
            reasons_text.append(item["reason"])

    if precomputed_embeddings is not None:
        embeddings = precomputed_embeddings
        print(f"Using precomputed embeddings, {embeddings.shape[0]} samples")
    else:
        print(f"Generating embeddings for {len(reasons_text)} texts...")
        embeddings = encode_texts(reasons_text, model_name=model_name, backend=backend)
    print(f"  Embedding dimension: {embeddings.shape}")

    raw_embeddings = embeddings.copy()

    pca_model = None
    if use_pca and embeddings.shape[1] > pca_components:
        print(f"Applying PCA to reduce to {pca_components} dimensions...")
        pca_model = PCA(n_components=pca_components, random_state=random_state)
        embeddings = pca_model.fit_transform(embeddings)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    print(f"\nK-Means clustering (k={n_clusters})...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)

    silhouette = None
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(embeddings, labels)
        print(f"  Silhouette Score: {silhouette:.4f}")
    else:
        print("  Silhouette Score: single cluster, cannot compute")

    return {
        "embeddings": embeddings,
        "raw_embeddings": raw_embeddings,
        "cluster_labels": labels,
        "cluster_centers": centers,
        "model_name": model_name,
        "encoder_backend": backend,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "normalize_text": normalize_text,
        "pca_model": pca_model,
        "pca_components": pca_components if pca_model else embeddings.shape[1]
    }


def analyze_clusters(reasons_data, clustering_result):
    """Extract representative samples and keywords for each cluster"""
    labels = clustering_result["cluster_labels"]
    embeddings = clustering_result["embeddings"]
    centers = clustering_result["cluster_centers"]
    n_clusters = clustering_result["n_clusters"]

    info = {}
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        reasons = [reasons_data[i]["reason"] for i in idx]
        normalized = [reasons_data[i].get("normalized_reason", reasons_data[i]["reason"].lower()) for i in idx]
        actions = [reasons_data[i]["action"] for i in idx]

        cluster_emb = embeddings[idx]
        centroid = centers[cid]
        distances = np.linalg.norm(cluster_emb - centroid, axis=1)
        repr_reason = reasons[int(np.argmin(distances))]

        words = []
        for norm_text in normalized:
            words.extend(norm_text.split())
        stopwords = {"the", "a", "an", "is", "are", "and", "of", "to", "in", "on", "at", "for", "it", "there", "as", "because"}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        keywords = [w.replace("_", " ") for w, _ in Counter(words).most_common(5)]

        info[cid] = {
            "size": len(idx),
            "percentage": round(len(idx) / len(reasons_data) * 100, 2),
            "keywords": ", ".join(keywords),
            "representative_reason": repr_reason,
            "sample_reasons": reasons[:3],
            "sample_actions": actions[:3]
        }

    return info


def summarize_clusters(cluster_info, keyword_groups=None):
    total_clusters = len(cluster_info)
    total_samples = sum(info["size"] for info in cluster_info.values())
    print("\n" + "=" * 70)
    print(f"Cluster overview: {total_clusters} clusters, {total_samples} samples")
    print("Top 10 largest clusters:")
    largest = sorted(cluster_info.items(), key=lambda kv: kv[1]["size"], reverse=True)[:10]
    for cid, info in largest:
        print(f"  #{cid:<3} size={info['size']:<5} ({info['percentage']:.2f}%) keywords={info['keywords']}")

    if keyword_groups:
        print("\nSemantic keyword aggregation:")
        for group, patterns in keyword_groups.items():
            matched_ids = []
            matched_size = 0
            for cid, info in cluster_info.items():
                keyword_text = info["keywords"].lower()
                if any(pattern in keyword_text for pattern in patterns):
                    matched_ids.append(cid)
                    matched_size += info["size"]
            if matched_ids:
                pct = matched_size / total_samples * 100
                ids_str = ",".join(str(i) for i in sorted(matched_ids))
                print(f"  - {group}: {matched_size} samples ({pct:.2f}%), clusters={ids_str}")
            else:
                print(f"  - {group}: no matching clusters")


def visualize_cluster_sizes(cluster_info, output_dir, highlight_keywords=None):
    if not PLOT_AVAILABLE:
        print("Plotting library not installed, skipping cluster size visualization")
        return

    highlight_keywords = highlight_keywords or []
    sorted_items = sorted(cluster_info.items(), key=lambda kv: kv[1]["size"], reverse=True)
    cluster_ids = [cid for cid, _ in sorted_items]
    sizes = [info["size"] for _, info in sorted_items]
    colors = []
    for _, info in sorted_items:
        kw = info["keywords"].lower()
        if any(hk in kw for hk in highlight_keywords):
            colors.append("tab:red")
        else:
            colors.append("tab:blue")

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(cluster_ids)), sizes, color=colors)
    plt.xlabel("Cluster (sorted by size)")
    plt.ylabel("Sample count")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    out_path = output_dir / "cluster_sizes.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved cluster size visualization: {out_path}")

def visualize_clusters(embeddings, labels, n_clusters, output_dir):
    if not PLOT_AVAILABLE:
        print("Plotting library not installed, skipping visualization")
        return

    print("\n" + "=" * 70)
    print("Generating t-SNE visualization")
    print("=" * 70)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab20", s=10, alpha=0.6)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"BDD-X Reasons Clustering (k={n_clusters})")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    out_path = output_dir / "clusters_tsne.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved visualization: {out_path}")


def find_optimal_k(
    embeddings,
    k_range=(20, 101, 5),
    sample_size=8000,
    random_state=RANDOM_SEED
):
    """Find optimal k using multiple metrics"""
    print("\n" + "=" * 70)
    print(f"Finding optimal k (range: {k_range[0]}-{k_range[1]-1}, step {k_range[2]})")
    print("Metrics: Inertia(lower) / Silhouette(higher) / Calinski-Harabasz(higher) / Davies-Bouldin(lower)")
    print("=" * 70)

    start, stop, step = k_range
    k_values = list(range(start, stop, step))

    if embeddings.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        eval_embeddings = embeddings[idx]
        print(f"  Using {sample_size} sampled samples for evaluation")
    else:
        eval_embeddings = embeddings

    results = []
    for k in k_values:
        print(f"\nTrying k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(eval_embeddings)

        inertia = kmeans.inertia_
        silhouette = None
        ch_score = None
        db_score = None
        if k > 1 and len(np.unique(labels)) > 1:
            silhouette = silhouette_score(eval_embeddings, labels)
            ch_score = calinski_harabasz_score(eval_embeddings, labels)
            db_score = davies_bouldin_score(eval_embeddings, labels)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
            "calinski_harabasz": ch_score,
            "davies_bouldin": db_score
        })

        print(f"  Inertia: {inertia:.2f}")
        if silhouette is not None:
            print(f"  Silhouette: {silhouette:.4f}")
            print(f"  Calinski-Harabasz: {ch_score:.2f}")
            print(f"  Davies-Bouldin: {db_score:.4f}")
        else:
            print("  Single cluster, skipping other metrics")

    vals_inertia = np.array([r["inertia"] for r in results], dtype=float)
    vals_sil = np.array([r["silhouette"] if r["silhouette"] is not None else np.nan for r in results], dtype=float)
    vals_ch = np.array([r["calinski_harabasz"] if r["calinski_harabasz"] is not None else np.nan for r in results], dtype=float)
    vals_db = np.array([r["davies_bouldin"] if r["davies_bouldin"] is not None else np.nan for r in results], dtype=float)

    def normalize(arr, higher_is_better=True):
        arr = arr.astype(float)
        mask = np.isnan(arr)
        if np.all(mask):
            return np.zeros_like(arr)
        valid = arr[~mask]
        if np.allclose(valid.max(), valid.min()):
            norm = np.zeros_like(arr)
        else:
            norm = (arr - valid.min()) / (valid.max() - valid.min())
        norm[mask] = 0.0
        return norm if higher_is_better else 1.0 - norm

    norm_inertia = normalize(vals_inertia, higher_is_better=False)
    norm_sil = normalize(vals_sil, higher_is_better=True)
    norm_ch = normalize(vals_ch, higher_is_better=True)
    norm_db = normalize(vals_db, higher_is_better=False)

    composite = 0.4 * norm_sil + 0.3 * norm_ch + 0.2 * norm_db + 0.1 * norm_inertia
    for i, r in enumerate(results):
        r["score"] = float(composite[i])

    best_idx = int(np.nanargmax(composite))
    recommended_k = results[best_idx]["k"]

    best_silhouette_k = results[int(np.nanargmax(norm_sil))]["k"] if not np.all(np.isnan(norm_sil)) else recommended_k
    best_ch_k = results[int(np.nanargmax(norm_ch))]["k"] if not np.all(np.isnan(norm_ch)) else recommended_k
    best_db_k = results[int(np.nanargmax(norm_db))]["k"] if not np.all(np.isnan(norm_db)) else recommended_k

    print("\n" + "=" * 70)
    print("Multi-metric evaluation results:")
    print("=" * 70)
    print(f"  Best Silhouette k = {best_silhouette_k}")
    print(f"  Best Calinski-Harabasz k = {best_ch_k}")
    print(f"  Best Davies-Bouldin k = {best_db_k}")
    print(f"  Recommended k = {recommended_k}")

    if PLOT_AVAILABLE:
        ks = [r["k"] for r in results]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(ks, vals_inertia, marker="o", label="Inertia (lower)", color="tab:blue")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ks, vals_sil, marker="^", label="Silhouette (higher)", color="tab:orange")
        ax2.plot(ks, vals_ch, marker="s", label="Calinski-Harabasz (higher)", color="tab:green")
        ax2.plot(ks, vals_db, marker="x", label="Davies-Bouldin (lower)", color="tab:red")
        ax2.set_ylabel("Scores")
        ax1.axvline(recommended_k, color="purple", linestyle="--", label=f"Recommended k={recommended_k}")
        handles, labels = [], []
        for ax in [ax1, ax2]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        fig.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "optimal_k_analysis.png", dpi=150)
        plt.close()
        print(f"  Saved metrics comparison plot: {OUTPUT_DIR / 'optimal_k_analysis.png'}")

    return {
        "results": results,
        "best_silhouette_k": best_silhouette_k,
        "best_calinski_harabasz_k": best_ch_k,
        "best_davies_bouldin_k": best_db_k,
        "recommended_k": recommended_k
    }


def merge_similar_clusters(
    embeddings,
    labels,
    centers,
    similarity_threshold=0.9,
    min_cluster_size=80,
    silhouette_before=None
):
    """Merge clusters with similar centers or small sample sizes"""
    unique_labels = np.sort(np.unique(labels))
    if len(unique_labels) <= 1:
        return labels, centers, {
            "merged": False,
            "n_original": len(unique_labels),
            "n_final": len(unique_labels),
            "label_mapping": {int(l): int(l) for l in unique_labels},
            "groups": {int(l): [int(l)] for l in unique_labels},
            "similarity_threshold": similarity_threshold,
            "min_cluster_size": min_cluster_size,
            "silhouette_before": silhouette_before,
            "silhouette_after": silhouette_before
        }

    size_map = {label: int(np.sum(labels == label)) for label in unique_labels}
    centers_ordered = np.vstack([centers[label] for label in unique_labels])
    sim_matrix = cosine_similarity(centers_ordered)

    n_clusters = len(unique_labels)
    parent = {i: i for i in range(n_clusters)}

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # 1) Merge clusters with highly similar centers
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if sim_matrix[i, j] >= similarity_threshold:
                union(i, j)

    # 2) Merge small clusters into nearest large clusters
    for idx, label in enumerate(unique_labels):
        if size_map[label] < min_cluster_size:
            sims = sim_matrix[idx].copy()
            sims[idx] = -np.inf
            nearest = np.argmax(sims)
            union(idx, nearest)

    root_to_newid = {}
    groups = defaultdict(list)
    new_id = 0

    for idx, label in enumerate(unique_labels):
        root = find(idx)
        if root not in root_to_newid:
            root_to_newid[root] = new_id
            new_id += 1
        gid = root_to_newid[root]
        groups[gid].append(int(label))

    mapping = {int(label): root_to_newid[find(idx)] for idx, label in enumerate(unique_labels)}
    new_labels = np.array([mapping[int(lbl)] for lbl in labels])

    new_centers = []
    for gid, old_labels in groups.items():
        mask = np.isin(labels, old_labels)
        cluster_emb = embeddings[mask]
        center = cluster_emb.mean(axis=0)
        center = center / np.clip(np.linalg.norm(center), 1e-12, None)
        new_centers.append(center)
    new_centers = np.vstack(new_centers)

    silhouette_after = None
    if len(np.unique(new_labels)) > 1 and embeddings.shape[0] > len(np.unique(new_labels)):
        silhouette_after = silhouette_score(embeddings, new_labels)

    merge_info = {
        "merged": len(np.unique(new_labels)) != len(unique_labels),
        "n_original": int(len(unique_labels)),
        "n_final": int(len(np.unique(new_labels))),
        "label_mapping": mapping,
        "groups": {int(k): v for k, v in groups.items()},
        "similarity_threshold": similarity_threshold,
        "min_cluster_size": min_cluster_size,
        "silhouette_before": silhouette_before,
        "silhouette_after": silhouette_after
    }
    return new_labels, new_centers, merge_info


def merge_clusters_by_keyword_groups(
    reasons_data,
    labels,
    embeddings,
    keyword_groups=FORCED_KEYWORD_GROUPS,
    silhouette_before=None,
    min_ratio=0.4,
    min_support=10
):
    """Force merge semantically identical clusters based on keyword ratio"""
    if not keyword_groups:
        return labels, None, {
            "merged": False,
            "n_original": int(len(np.unique(labels))),
            "n_final": int(len(np.unique(labels))),
            "groups": {},
            "silhouette_before": silhouette_before,
            "silhouette_after": silhouette_before
        }

    unique_labels = np.sort(np.unique(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    parent = {idx: idx for idx in range(len(unique_labels))}

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    cluster_text = {}
    cluster_counts = {}
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        texts = [reasons_data[i].get("normalized_reason", reasons_data[i]["reason"].lower()) for i in idxs]
        cluster_text[label] = texts
        cluster_counts[label] = len(texts)

    for group_name, patterns in keyword_groups.items():
        matched = []
        for label in unique_labels:
            texts = cluster_text[label]
            if not texts:
                continue
            hits = sum(1 for t in texts if any(pattern in t for pattern in patterns))
            if hits >= min_support and hits / len(texts) >= min_ratio:
                matched.append(label)
        if len(matched) > 1:
            base_idx = label_to_idx[matched[0]]
            for other in matched[1:]:
                union(base_idx, label_to_idx[other])

    # Build new labels
    root_to_newid = {}
    groups = defaultdict(list)
    next_id = 0
    for label in unique_labels:
        root = find(label_to_idx[label])
        if root not in root_to_newid:
            root_to_newid[root] = next_id
            next_id += 1
        new_id = root_to_newid[root]
        groups[new_id].append(int(label))

    if next_id == len(unique_labels):
        return labels, None, {
            "merged": False,
            "n_original": int(len(unique_labels)),
            "n_final": int(len(unique_labels)),
            "groups": {int(k): v for k, v in groups.items()},
            "silhouette_before": silhouette_before,
            "silhouette_after": silhouette_before
        }

    mapping = {int(label): int(root_to_newid[find(label_to_idx[label])]) for label in unique_labels}
    new_labels = np.array([mapping[int(lbl)] for lbl in labels])

    # Recompute cluster centers
    new_centers = []
    for new_id, old_labels in groups.items():
        mask = np.isin(labels, old_labels)
        cluster_emb = embeddings[mask]
        center = cluster_emb.mean(axis=0)
        center = center / np.clip(np.linalg.norm(center), 1e-12, None)
        new_centers.append(center)
    new_centers = np.vstack(new_centers)

    silhouette_after = None
    if len(np.unique(new_labels)) > 1 and embeddings.shape[0] > len(np.unique(new_labels)):
        silhouette_after = silhouette_score(embeddings, new_labels)

    info = {
        "merged": True,
        "n_original": int(len(unique_labels)),
        "n_final": int(len(np.unique(new_labels))),
        "groups": {int(k): v for k, v in groups.items()},
        "silhouette_before": silhouette_before,
        "silhouette_after": silhouette_after,
        "min_ratio": min_ratio,
        "min_support": min_support
    }
    return new_labels, new_centers, info


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("BDD-X Reasons Semantic Clustering")
    print("=" * 70)

    print("\nStep 1: Loading data")
    reasons_data = load_all_reasons(split="all")

    normalize_option = True

    encoder_choice_input = input("Select text encoder (sbert/e5, default sbert): ").strip().lower()
    if encoder_choice_input not in {"", "sbert", "e5"}:
        print("Invalid choice, using default sbert")
        encoder_choice_input = ""
    encoder_choice = encoder_choice_input if encoder_choice_input else "sbert"

    default_model = "intfloat/e5-base-v2" if encoder_choice == "e5" else "all-MiniLM-L6-v2"
    model_name_input = input(f"Model name (default {default_model}): ").strip()
    model_name = model_name_input if model_name_input else default_model

    precomputed_embeddings = None

    use_optimal = input("\nPerform k value analysis? (y/n, default n): ").strip().lower()
    if use_optimal == "y":
        print("\nUsing all samples for k analysis (range 20-100, step 5)...")
        eval_texts = [normalize_reason_text(item["reason"]) for item in reasons_data] if normalize_option else [item["reason"] for item in reasons_data]
        print("Generating text embeddings for evaluation...")
        embeddings = encode_texts(eval_texts, model_name=model_name, backend=encoder_choice)
        precomputed_embeddings = embeddings
        k_info = find_optimal_k(embeddings, k_range=(20, 101, 5))
        with open(OUTPUT_DIR / "k_analysis.json", "w") as f:
            json.dump(k_info, f, indent=2)
        k_default = k_info["recommended_k"]
        use_default = input(f"\nRecommended k={k_default}, use it? (y/n, default y): ").strip().lower()
        if use_default == "n":
            n_clusters = int(input("Enter your k: "))
        else:
            n_clusters = k_default
    else:
        n_clusters = 35

    print(f"\nStep 2: Clustering (k={n_clusters}, encoder={encoder_choice}, model={model_name})")
    clustering_result = cluster_with_embeddings(
        reasons_data,
        model_name=model_name,
        backend=encoder_choice,
        n_clusters=n_clusters,
        normalize_text=normalize_option,
        precomputed_embeddings=precomputed_embeddings
    )

    print("\n" + "=" * 70)
    merge_choice = input("Perform semantic merging? (y/n, default y): ").strip().lower()
    if merge_choice != "n":
        threshold_input = input("Cosine similarity threshold (0.80-0.95, default 0.80): ").strip()
        try:
            sim_threshold = float(threshold_input) if threshold_input else 0.80
            sim_threshold = min(0.95, max(0.80, sim_threshold))
        except ValueError:
            sim_threshold = 0.80

        min_size_input = input("Minimum cluster size (default 200): ").strip()
        try:
            min_cluster_size = int(min_size_input) if min_size_input else 200
            min_cluster_size = max(10, min_cluster_size)
        except ValueError:
            min_cluster_size = 200

        new_labels, new_centers, merge_info = merge_similar_clusters(
            clustering_result["embeddings"],
            clustering_result["cluster_labels"],
            clustering_result["cluster_centers"],
            similarity_threshold=sim_threshold,
            min_cluster_size=min_cluster_size,
            silhouette_before=clustering_result.get("silhouette_score")
        )

        if merge_info["merged"]:
            clustering_result["cluster_labels"] = new_labels
            clustering_result["cluster_centers"] = new_centers
            clustering_result["n_clusters"] = merge_info["n_final"]
            if merge_info["silhouette_after"] is not None:
                clustering_result["silhouette_score"] = merge_info["silhouette_after"]

            merge_info_file = OUTPUT_DIR / "merge_info.json"
            with open(merge_info_file, "w") as f:
                json.dump({
                    "merged": merge_info["merged"],
                    "n_original": merge_info["n_original"],
                    "n_final": merge_info["n_final"],
                    "label_mapping": {int(k): int(v) for k, v in merge_info["label_mapping"].items()},
                    "groups": {int(k): [int(x) for x in v] for k, v in merge_info["groups"].items()},
                    "similarity_threshold": merge_info["similarity_threshold"],
                    "min_cluster_size": merge_info["min_cluster_size"],
                    "silhouette_before": merge_info["silhouette_before"],
                    "silhouette_after": merge_info["silhouette_after"]
                }, f, indent=2)
            print(f"\nSaved merge info: {merge_info_file}")
            if merge_info["silhouette_before"] is not None and merge_info["silhouette_after"] is not None:
                print(f"  Silhouette: {merge_info['silhouette_before']:.4f} -> {merge_info['silhouette_after']:.4f}")
        else:
            print("No merge triggered, keeping original cluster assignment.")

    # Keyword-driven forced merge
    print("\nPerforming keyword-driven merge check...")
    ratio_input = input("Keyword merge minimum ratio (0-1, default 0.4): ").strip()
    support_input = input("Keyword match minimum sample count (default 10): ").strip()
    try:
        kw_min_ratio = float(ratio_input) if ratio_input else 0.4
        kw_min_ratio = min(1.0, max(0.1, kw_min_ratio))
    except ValueError:
        kw_min_ratio = 0.4
    try:
        kw_min_support = int(support_input) if support_input else 10
        kw_min_support = max(1, kw_min_support)
    except ValueError:
        kw_min_support = 10

    kw_labels, kw_centers, kw_info = merge_clusters_by_keyword_groups(
        reasons_data,
        clustering_result["cluster_labels"],
        clustering_result["embeddings"],
        keyword_groups=FORCED_KEYWORD_GROUPS,
        silhouette_before=clustering_result.get("silhouette_score"),
        min_ratio=kw_min_ratio,
        min_support=kw_min_support
    )
    if kw_info["merged"]:
        clustering_result["cluster_labels"] = kw_labels
        clustering_result["cluster_centers"] = kw_centers
        clustering_result["n_clusters"] = kw_info["n_final"]
        if kw_info["silhouette_after"] is not None:
            clustering_result["silhouette_score"] = kw_info["silhouette_after"]

        kw_info_file = OUTPUT_DIR / "keyword_merge_info.json"
        with open(kw_info_file, "w") as f:
            json.dump({
                "groups": {int(k): [int(x) for x in v] for k, v in kw_info["groups"].items()},
                "n_original": kw_info["n_original"],
                "n_final": kw_info["n_final"],
                "silhouette_before": kw_info["silhouette_before"],
                "silhouette_after": kw_info["silhouette_after"],
                "min_ratio": kw_info.get("min_ratio"),
                "min_support": kw_info.get("min_support")
            }, f, indent=2)
        print(f"Keyword merge applied, saved info: {kw_info_file}")
        if kw_info["silhouette_before"] is not None and kw_info["silhouette_after"] is not None:
            print(f"  Silhouette: {kw_info['silhouette_before']:.4f} -> {kw_info['silhouette_after']:.4f}")
    else:
        print("Keyword merge found no clusters to process.")

    print("\nStep 3: Cluster analysis")
    cluster_info = analyze_clusters(reasons_data, clustering_result)
    summarize_clusters(cluster_info, FORCED_KEYWORD_GROUPS)

    print(f"\nEncoder: {clustering_result.get('encoder_backend', 'unknown')} | Model: {clustering_result.get('model_name', '')}")
    print("\n" + "=" * 70)
    print("Cluster Summary")
    print("=" * 70)
    for cid in sorted(cluster_info.keys()):
        info = cluster_info[cid]
        print(f"\nCluster {cid}: {info['size']} samples ({info['percentage']:.2f}%)")
        print(f"   Keywords: {info['keywords']}")
        print(f"   Representative: {info['representative_reason']}")
        print("   Sample reasons:")
        for i, reason in enumerate(info["sample_reasons"][:2], 1):
            print(f"     {i}. {reason}")
        print("   Sample actions:")
        for i, action in enumerate(info["sample_actions"][:2], 1):
            print(f"     {i}. {action}")

    visualize_cluster_sizes(cluster_info, OUTPUT_DIR, highlight_keywords=["traffic light red", "red light"])

    print("\nStep 4: Saving results")
    labels_file = OUTPUT_DIR / "cluster_labels.npy"
    np.save(labels_file, clustering_result["cluster_labels"])
    print(f"Saved cluster labels: {labels_file}")

    emb_file = OUTPUT_DIR / "embeddings.npy"
    np.save(emb_file, clustering_result["embeddings"])
    print(f"Saved clustering embeddings: {emb_file}")

    if "raw_embeddings" in clustering_result:
        raw_emb_file = OUTPUT_DIR / "embeddings_raw.npy"
        np.save(raw_emb_file, clustering_result["raw_embeddings"])
        print(f"Saved raw S-BERT embeddings: {raw_emb_file}")

    analysis_file = OUTPUT_DIR / "cluster_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(cluster_info, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster analysis: {analysis_file}")

    for i, label in enumerate(clustering_result["cluster_labels"]):
        reasons_data[i]["cluster"] = int(label)
    clustered_data_file = OUTPUT_DIR / "reasons_with_clusters.json"
    with open(clustered_data_file, "w", encoding="utf-8") as f:
        json.dump(reasons_data, f, indent=2, ensure_ascii=False)
    print(f"Saved labeled data: {clustered_data_file}")

    print("\nStep 5: Visualization")
    visualize_clusters(
        clustering_result["embeddings"],
        clustering_result["cluster_labels"],
        clustering_result["n_clusters"],
        OUTPUT_DIR
    )

    print("\n" + "=" * 70)
    print("Clustering complete!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  cluster_analysis.json - Statistics for each cluster")
    print("  cluster_labels.npy - Cluster labels")
    print("  embeddings.npy - Normalized embeddings")
    if "raw_embeddings" in clustering_result:
        print("  embeddings_raw.npy - Raw embeddings")
    print("  reasons_with_clusters.json - Original data with cluster labels")
    if PLOT_AVAILABLE:
        print("  clusters_tsne.png - t-SNE visualization")
        print("  optimal_k_analysis.png - Metrics comparison plot")


if __name__ == "__main__":
    main()
