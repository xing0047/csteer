from behaviors import ALL_BEHAVIORS, get_vector_path
import torch as t
import os
import argparse
from behaviors import NORMALIZED_VECTORS_PATH

def normalize_vectors(model_name: str, model_size: str, vector_suffix: str, layers: list):
    # make normalized_vectors directory
    normalized_vectors_dir = NORMALIZED_VECTORS_PATH + '/' + "refer" + '/' + model_name + "_" + model_size
    os.makedirs(normalized_vectors_dir, exist_ok=True)
    for layer in layers:
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in ALL_BEHAVIORS:
            vec_path = get_vector_path(behavior, model_name, model_size, vector_suffix, layer)
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("VECTOR", "NORMALIZED_VECTOR")
            new_paths[behavior] = new_path
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in ALL_BEHAVIORS:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in ALL_BEHAVIORS:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, choices=["internvl3_5", "qwen3vl"]
    )
    parser.add_argument(
        "--vector_suffix", type=str, required=True
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=list(range(28))
    )
    args = parser.parse_args()
    normalize_vectors(args.model_name, "8b", args.vector_suffix, args.layers)
    print(f"Vectors saved locally to: ../NORMALIZED_VECTORS")
