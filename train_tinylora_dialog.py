import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

from my_model import TransformerModel, MAX_LEN, MODEL_CONFIGS, MODEL_SIZE


def build_arg_parser():
    p = argparse.ArgumentParser(description="TinyLoRA-style dialog fine-tuning (SFT).")
    p.add_argument("--base-model-path", type=str, default="transformer_ttt_personachat_hf.pth")
    p.add_argument("--tokenizer-path", type=str, default="./tokenizer_hf")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--dataset", type=str, default="OpenRL/daily_dialog")
    p.add_argument("--max-pairs", type=int, default=2000000)
    p.add_argument("--save-path", type=str, default="tinylora_dialog_adapter.pth")
    return p


class SharedTinyLoRA(nn.Module):
    def __init__(self, out_dim, in_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(out_dim) * 0.01)
        self.v = nn.Parameter(torch.randn(in_dim) * 0.01)


class TinyLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, shared: SharedTinyLoRA):
        super().__init__()
        self.base = base
        self.shared = shared
        self.scale = nn.Parameter(torch.zeros(1))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        proj = torch.einsum("...d,d->...", x, self.shared.v)
        delta = proj.unsqueeze(-1) * self.shared.u
        return y + self.scale * delta


def inject_tinylora(model: nn.Module, target_keywords=("wq", "wk", "wv", "wo", "w1", "w2", "w3")):
    shared_registry = {}

    def get_shared(out_dim, in_dim):
        key = (out_dim, in_dim)
        if key not in shared_registry:
            shared_registry[key] = SharedTinyLoRA(out_dim, in_dim)
        return shared_registry[key]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not any(k in name for k in target_keywords):
                continue
            parent = model
            if "." in name:
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                child_name = parts[-1]
            else:
                child_name = name
            shared = get_shared(module.out_features, module.in_features)
            setattr(parent, child_name, TinyLoRALinear(module, shared))

    return model


def freeze_base_params(model: nn.Module):
    for n, p in model.named_parameters():
        if "shared" in n or "scale" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def move_tinylora_to_device(model: nn.Module, device: torch.device):
    for module in model.modules():
        if isinstance(module, TinyLoRALinear):
            module.shared.u.data = module.shared.u.data.to(device)
            module.shared.v.data = module.shared.v.data.to(device)
            module.scale.data = module.scale.data.to(device)


def clean_persona_chat(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dialog_lines = []
    for ln in lines:
        parts = ln.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            content = parts[1].strip()
        else:
            content = ln
        if "persona:" in content:
            continue
        dialog_lines.append(content)
    dialog_text = " ".join(dialog_lines)
    utterances = [u.strip() for u in dialog_text.split("|") if u.strip()]
    return "\n".join(utterances) if utterances else dialog_text


def main():
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_SIZE: {MODEL_SIZE}. Options: {list(MODEL_CONFIGS.keys())}")
    num_blocks, dim, num_heads = MODEL_CONFIGS[MODEL_SIZE]

    model = TransformerModel(
        len(tokenizer),
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        window_size=args.max_len,
        max_seq_len=args.max_len,
        tie_word_embeddings=False,
    ).to(device)

    if os.path.exists(args.base_model_path):
        state = torch.load(args.base_model_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model = inject_tinylora(model)
    freeze_base_params(model)
    move_tinylora_to_device(model, device)
    model.train()

    dataset = load_dataset(args.dataset)

    col_names = dataset["train"].column_names
    if "dialog" in col_names:
        dialog_col = "dialog"
    elif "dialogue" in col_names:
        dialog_col = "dialogue"
    elif "utterances" in col_names:
        dialog_col = "utterances"
    elif "text" in col_names:
        dialog_col = "text"
    else:
        raise ValueError(f"Unknown dialog column. Columns: {col_names}")

    pairs = []
    for item in tqdm(dataset["train"], desc="Building pairs"):
        dialog = item[dialog_col]
        if isinstance(dialog, dict) and "dialog" in dialog:
            dialog = dialog["dialog"]
        for i in range(len(dialog) - 1):
            user = dialog[i].strip()
            assistant = dialog[i + 1].strip()
            if not user or not assistant:
                continue
            pairs.append(f"User: {user}\nAssistant: {assistant}")
            if len(pairs) >= args.max_pairs:
                break
        if len(pairs) >= args.max_pairs:
            break

    tokenized = tokenizer(
        pairs,
        truncation=True,
        padding="max_length",
        max_length=args.max_len,
    )["input_ids"]

    inputs = torch.tensor(tokenized, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs), batch_size=args.batch, shuffle=True)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, fused=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            logits = model(batch)
            # shift for next-token prediction
            logits = logits[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            steps += 1
        avg = total_loss / max(steps, 1)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

    # Save only adapter params
    adapter_state = {k: v for k, v in model.state_dict().items() if "shared" in k or "scale" in k}
    torch.save(adapter_state, args.save_path)
    print(f"Saved TinyLoRA adapters to: {args.save_path}")


if __name__ == "__main__":
    main()
