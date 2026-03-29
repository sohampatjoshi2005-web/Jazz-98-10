import os, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from evaluate import load as load_metric

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "bert-base-uncased"
MAX_LENGTH  = 384
STRIDE      = 128
BATCH_SIZE  = 8
EPOCHS      = 2
LR          = 2e-5
SQUAD_TRAIN = 800
SQUAD_VAL   = 200
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_squad_subset():
    dataset = load_dataset("squad", split="validation")
    return dataset.select(range(SQUAD_TRAIN)), dataset.select(range(SQUAD_TRAIN, SQUAD_TRAIN + SQUAD_VAL))


class SQuADDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=MAX_LENGTH, stride=STRIDE, is_train=True):
        self.examples = []
        self.is_train = is_train
        for item in hf_dataset:
            question = item["question"]
            context  = item["context"]
            answers  = item["answers"]
            enc = tokenizer(
                question, context,
                max_length=max_length, truncation="only_second", stride=stride,
                return_overflowing_tokens=True, return_offsets_mapping=True,
                padding="max_length", return_tensors="pt",
            )
            offset_mapping = enc.pop("offset_mapping")
            enc.pop("overflow_to_sample_mapping", None)
            for i in range(enc["input_ids"].shape[0]):
                offsets = offset_mapping[i]
                seq_ids = enc.sequence_ids(i)
                if is_train:
                    if not answers["text"]:
                        start_pos, end_pos = 0, 0
                    else:
                        ans_text  = answers["text"][0]
                        ans_start = answers["answer_start"][0]
                        ans_end   = ans_start + len(ans_text)
                        ctx_start = next((j for j, s in enumerate(seq_ids) if s == 1), None)
                        ctx_end   = next((len(seq_ids)-1-j for j, s in enumerate(reversed(seq_ids)) if s == 1), None)
                        if offsets[ctx_start][0] > ans_end or offsets[ctx_end][1] < ans_start:
                            start_pos, end_pos = 0, 0
                        else:
                            tok = ctx_start
                            while tok <= ctx_end and offsets[tok][0] <= ans_start:
                                tok += 1
                            start_pos = tok - 1
                            tok = ctx_end
                            while tok >= ctx_start and offsets[tok][1] >= ans_end:
                                tok -= 1
                            end_pos = tok + 1
                    self.examples.append({
                        "input_ids":       enc["input_ids"][i],
                        "attention_mask":  enc["attention_mask"][i],
                        "token_type_ids":  enc["token_type_ids"][i],
                        "start_positions": torch.tensor(start_pos),
                        "end_positions":   torch.tensor(end_pos),
                    })
                else:
                    self.examples.append({
                        "input_ids":      enc["input_ids"][i],
                        "attention_mask": enc["attention_mask"][i],
                        "token_type_ids": enc["token_type_ids"][i],
                        "offset_mapping": offsets,
                        "context":        context,
                        "question":       question,
                        "answer_text":    answers["text"][0] if answers["text"] else "",
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_train(batch):
    return {
        "input_ids":       torch.stack([b["input_ids"]       for b in batch]),
        "attention_mask":  torch.stack([b["attention_mask"]  for b in batch]),
        "token_type_ids":  torch.stack([b["token_type_ids"]  for b in batch]),
        "start_positions": torch.stack([b["start_positions"] for b in batch]),
        "end_positions":   torch.stack([b["end_positions"]   for b in batch]),
    }

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(train_dataset):
    model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch+1} Step {step}/{len(loader)}  loss={loss.item():.4f}")
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")
    return model, time.time() - t0

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, val_raw):
    model.eval()
    predictions, references = [], []
    for item in val_raw:
        enc = tokenizer(
            item["question"], item["context"],
            max_length=MAX_LENGTH, truncation="only_second",
            return_offsets_mapping=True, return_tensors="pt", padding="max_length",
        )
        offset_mapping = enc.pop("offset_mapping")[0]
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        start_idx = out.start_logits[0].argmax().item()
        end_idx   = max(out.end_logits[0].argmax().item(), start_idx)
        start_char = offset_mapping[start_idx][0].item()
        end_char   = offset_mapping[end_idx][1].item()
        predictions.append({"id": item["id"], "prediction_text": item["context"][start_char:end_char]})
        references.append({"id": item["id"], "answers": item["answers"]})
    return load_metric("squad").compute(predictions=predictions, references=references)

# ── Interpretability ──────────────────────────────────────────────────────────
def get_attention_weights(model, tokenizer, question, context):
    model.eval()
    enc = tokenizer(question, context, max_length=MAX_LENGTH, truncation="only_second",
                    return_tensors="pt", padding="max_length")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attentions = [a.squeeze(0).cpu().numpy() for a in out.attentions]
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return attentions, tokens, enc, out


def get_gradient_importance(model, tokenizer, question, context):
    model.eval()
    enc = tokenizer(question, context, max_length=MAX_LENGTH, truncation="only_second",
                    return_tensors="pt", padding="max_length")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    embeddings = model.bert.embeddings(input_ids=enc["input_ids"], token_type_ids=enc["token_type_ids"])
    embeddings.retain_grad()
    out = model(inputs_embeds=embeddings, attention_mask=enc["attention_mask"], token_type_ids=enc["token_type_ids"])
    score = out.start_logits[0].max() + out.end_logits[0].max()
    model.zero_grad()
    score.backward()
    importance = (embeddings.grad[0] * embeddings[0].detach()).abs().sum(dim=-1)
    importance = importance.cpu().detach().numpy()
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return importance, tokens

# ── Visualizations ────────────────────────────────────────────────────────────
def find_sep(tokens):
    for i, t in enumerate(tokens):
        if t == "[SEP]":
            return i
    return len(tokens) // 2


def plot_token_importance_heatmap(grad_scores, attn_scores, tokens, title_prefix=""):
    sep = find_sep(tokens)
    n = min(40, len(tokens))
    tok_short = [t[:8] for t in tokens[:n]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, scores, label, cmap in zip(
        axes,
        [grad_scores[:n], attn_scores[:n]],
        ["Gradient x Input Importance", "Mean Attention Score"],
        ["YlOrRd", "Blues"],
    ):
        data = scores.reshape(1, -1)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(tok_short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([])
        if sep < n:
            ax.axvline(sep - 0.5, color="black", linewidth=1.5, linestyle="--")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle(f"{title_prefix} - Token Importance Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{title_prefix.replace(' ', '_')}_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_attention_head_grid(attentions, tokens, layer=11, n_show=8):
    n = min(30, len(tokens))
    tok_short = [t[:6] for t in tokens[:n]]
    heads = attentions[layer][:n_show, :n, :n]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for h, ax in enumerate(axes[:n_show]):
        im = ax.imshow(heads[h], cmap="viridis", vmin=0, vmax=heads[h].max())
        ax.set_title(f"Head {h+1}", fontsize=10)
        ax.set_xticks(range(n))
        ax.set_xticklabels(tok_short, rotation=90, fontsize=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(tok_short, fontsize=5)
    fig.suptitle(f"Attention Heads - Layer {layer+1}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "attention_head_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_layer_mean_attention(attentions, tokens):
    n_layers = len(attentions)
    n_tokens = min(30, attentions[0].shape[-1])
    tok_short = [t[:8] for t in tokens[:n_tokens]]
    layer_attn = np.array([attn.mean(axis=0)[0, :n_tokens] for attn in attentions])
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(layer_attn, aspect="auto", cmap="plasma")
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tok_short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i+1}" for i in range(n_layers)], fontsize=8)
    ax.set_title("CLS Token Attention Across Layers", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "layer_mean_attention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_comparison_bar(grad_scores, attn_scores, tokens):
    n = min(25, len(tokens))
    tok_short = [t[:10] for t in tokens[:n]]
    x = np.arange(n)
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, grad_scores[:n], width, label="Gradient Importance", color="#534AB7", alpha=0.85)
    ax.bar(x + width/2, attn_scores[:n], width, label="Attention Score",      color="#1D9E75", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tok_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Score")
    ax.set_title("Gradient vs Attention - Per Token Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    sep = find_sep(tokens)
    if sep < n:
        ax.axvline(sep - 0.5, color="black", linewidth=1.2, linestyle="--")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gradient_vs_attention_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def bias_analysis(model, tokenizer, bias_pairs):
    results = {}
    model.eval()
    for label, (q, ctx_f, ctx_m) in bias_pairs.items():
        scores = []
        for ctx in [ctx_f, ctx_m]:
            enc = tokenizer(q, ctx, return_tensors="pt", max_length=MAX_LENGTH,
                            truncation="only_second", padding="max_length")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_attentions=True)
            attn = out.attentions[-1][0, 0, 0, :]
            scores.append(attn.mean().item())
        results[label] = scores
    return results


def plot_bias(bias_results):
    labels = list(bias_results.keys())
    female = [v[0] for v in bias_results.values()]
    male   = [v[1] for v in bias_results.values()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, female, 0.4, label="Female pronoun context", color="#D4537E", alpha=0.85)
    ax.bar(x + 0.2, male,   0.4, label="Male pronoun context",   color="#185FA5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean CLS Attention")
    ax.set_title("Bias Analysis: Mean CLS Attention by Gender Context", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def save_results(results):
    path = os.path.join(OUTPUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    print(json.dumps(results, indent=2))

# ── Main ──────────────────────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

print("\n=== Loading SQuAD Subset ===")
train_raw, val_raw = load_squad_subset()
print(f"Train: {len(train_raw)} | Val: {len(val_raw)}")

print("\n=== Preparing Dataset ===")
train_dataset = SQuADDataset(train_raw, tokenizer, is_train=True)
print(f"Train examples (with overflow): {len(train_dataset)}")

print("\n=== Baseline BERT (No Fine-tuning) ===")
baseline_model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
baseline_results = evaluate_model(baseline_model, tokenizer, val_raw)
baseline_params  = count_params(baseline_model)
print(f"Baseline  EM={baseline_results['exact_match']:.2f}  F1={baseline_results['f1']:.2f}")

print("\n=== Fine-tuning BERT ===")
ft_model, train_time = train_model(train_dataset)
ft_results = evaluate_model(ft_model, tokenizer, val_raw)
ft_params  = count_params(ft_model)
print(f"Fine-tuned  EM={ft_results['exact_match']:.2f}  F1={ft_results['f1']:.2f}  Time={train_time:.1f}s")

print("\n=== Interpretability Analysis ===")
sample   = val_raw[5]
question = sample["question"]
context  = sample["context"][:512]
print(f"Q: {question}")
print(f"A: {sample['answers']['text'][0] if sample['answers']['text'] else 'N/A'}")

attentions, tokens, enc, out = get_attention_weights(ft_model, tokenizer, question, context)

mean_attn = np.mean([a.mean(axis=0)[0, :] for a in attentions], axis=0)
mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min() + 1e-9)

grad_scores, _ = get_gradient_importance(ft_model, tokenizer, question, context)

print("\n--- Generating Visualizations ---")
plot_token_importance_heatmap(grad_scores, mean_attn, tokens, "Sample QA")
plot_attention_head_grid(attentions, tokens, layer=11)
plot_layer_mean_attention(attentions, tokens)
plot_comparison_bar(grad_scores, mean_attn, tokens)

bias_pairs = {
    "Doctor": (
        "Who treated the patient?",
        "She is a brilliant doctor who treated the patient carefully.",
        "He is a brilliant doctor who treated the patient carefully.",
    ),
    "Engineer": (
        "Who built the bridge?",
        "She is an engineer who built the bridge.",
        "He is an engineer who built the bridge.",
    ),
    "Nurse": (
        "Who cared for the patient?",
        "She is a compassionate nurse who cared for the patient.",
        "He is a compassionate nurse who cared for the patient.",
    ),
}
bias_results = bias_analysis(ft_model, tokenizer, bias_pairs)
plot_bias(bias_results)

all_results = {
    "baseline":   {"exact_match": round(baseline_results["exact_match"], 2), "f1": round(baseline_results["f1"], 2), "training_time_s": 0, "parameters_M": round(baseline_params, 1)},
    "fine_tuned": {"exact_match": round(ft_results["exact_match"], 2),       "f1": round(ft_results["f1"], 2),       "training_time_s": round(train_time, 1), "parameters_M": round(ft_params, 1)},
    "bias_analysis": {k: {"female": round(v[0], 5), "male": round(v[1], 5)} for k, v in bias_results.items()},
}
save_results(all_results)
print("\nDone. All outputs saved to ./outputs/")
