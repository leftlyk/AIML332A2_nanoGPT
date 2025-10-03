import torch
import matplotlib.pyplot as plt
import os

def plot_step_probs(probs, idx_candidates, idx_next, step, decoder, out_dir="out"):
    # prob for each sampled candidate
    prob_candidates = probs[idx_candidates]  # tensor of shape (10,)

    # idx_next should be scalar
    idx_next_scalar = idx_next.item()

    # colors: red = chosen token
    colors = ["tab:red" if idx.item() == idx_next_scalar else "tab:blue"
              for idx in idx_candidates]

    decoded_labels = [decoder([idx.item()]) for idx in idx_candidates]

    plt.figure(figsize=(8,5))
    plt.bar(range(len(idx_candidates)), prob_candidates.numpy(), color=colors)
    plt.xticks(range(len(idx_candidates)), decoded_labels, rotation=45, ha="right")
    plt.xlabel("Sampled tokens")
    plt.ylabel("Probability")
    plt.title(f"Step {step+1}: Model samples (red = chosen)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sampled_probs_step{step+1}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


