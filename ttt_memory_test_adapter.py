import argparse
import random
import torch
from transformers import AutoTokenizer

from my_model import TransformerModel, MAX_LEN, MODEL_CONFIGS, MODEL_SIZE
from train_tinylora_dialog import SharedTinyLoRA, TinyLoRALinear


def build_arg_parser():
    p = argparse.ArgumentParser(description="Full TTT memory test: base vs adapter.")
    p.add_argument("--model-path", type=str, default="transformer_ttt_personachat_hf.pth")
    p.add_argument("--adapter-path", type=str, default="tinylora_dialog_adapter.pth")
    p.add_argument("--tokenizer-path", type=str, default="./tokenizer_hf")
    p.add_argument("--window", type=int, default=MAX_LEN)
    p.add_argument("--ttt-batch", type=int, default=MAX_LEN)
    p.add_argument("--adapt-ratio", type=float, default=0.95)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--grid", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def inject_tinylora(model: torch.nn.Module, adapter_state):
    shared_registry = {}

    def get_shared(out_dim, in_dim):
        key = (out_dim, in_dim)
        if key not in shared_registry:
            shared_registry[key] = SharedTinyLoRA(out_dim, in_dim)
        return shared_registry[key]

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not any(k in name for k in ("wq", "wk", "wv", "wo", "w1", "w2", "w3")):
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

    model.load_state_dict(adapter_state, strict=False)
    return model


def build_model(tokenizer, window, model_path, adapter_path=None, device="cpu"):
    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_SIZE: {MODEL_SIZE}. Options: {list(MODEL_CONFIGS.keys())}")
    num_blocks, dim, num_heads = MODEL_CONFIGS[MODEL_SIZE]
    vocab_size = len(tokenizer)

    model = TransformerModel(
        vocab_size,
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        window_size=window,
        max_seq_len=MAX_LEN,
        tie_word_embeddings=False,
    ).to(device)

    base_state = torch.load(model_path, map_location=device)
    model.load_state_dict(base_state, strict=False)

    if adapter_path is not None:
        adapter_state = torch.load(adapter_path, map_location=device)
        model = inject_tinylora(model, adapter_state).to(device)

    model.eval()
    return model


def run_test(model, tokenizer, prompts, repeat, adapt_ratio, window, ttt_batch):
    pre_losses = []
    post_losses = []
    for pattern in prompts:
        text = (" " + pattern).join([""] * repeat)
        tokens = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=MAX_LEN,
            return_tensors="pt",
        )["input_ids"].to(next(model.parameters()).device)

        B, L = tokens.shape
        if L < 4:
            raise ValueError("Sequence too short for test.")

        adapt_end = max(1, int((L - 1) * adapt_ratio))
        adapt_end = min(adapt_end, L - 2)

        with torch.no_grad():
            fast_deltas = model._init_fast_deltas(B)
            pre_loss = model._post_ttt_loss(tokens, adapt_end, L - 1, fast_deltas)

        fast_deltas = model._init_fast_deltas(B)
        for j in range(0, adapt_end, ttt_batch):
            batch_end = min(j + ttt_batch, adapt_end)
            fast_deltas, _, _ = model._ttt_update(
                tokens, j, batch_end, window, fast_deltas
            )

        with torch.no_grad():
            post_loss = model._post_ttt_loss(tokens, adapt_end, L - 1, fast_deltas)

        pre_losses.append(pre_loss.item())
        post_losses.append(post_loss.item())

    pre_avg = sum(pre_losses) / len(pre_losses)
    post_avg = sum(post_losses) / len(post_losses)
    delta = pre_avg - post_avg
    pct = (delta / pre_avg) * 100.0
    return pre_avg, post_avg, delta, pct


def main():
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "I work as a software developer and write Python code daily.",
        "I live in a small apartment with a cat and two plants.",
        "I prefer coffee in the morning and tea in the evening.",
        "I usually go to the gym on Mondays and Fridays.",
        "I love hiking in the mountains during summer.",
        "My favorite movie is The Matrix, but I also like comedies.",
        "I am learning Spanish and practice with flashcards.",
        "On weekends I read science fiction and play chess.",
        "I commute by train and listen to podcasts.",
        "I have a blue backpack and a silver laptop.",
        "I enjoy cooking pasta and trying new recipes.",
        "I sleep about seven hours and wake up early.",
        "I keep a journal and write notes every evening.",
        "I like electronic music, but I also enjoy jazz.",
        "I sometimes work late when deadlines are close.",
        "I prefer cold weather to hot weather.",
        "I have a habit of taking short walks after lunch.",
        "I used to play soccer in school.",
        "I am saving money for a trip next year.",
        "I like to watch documentaries about space.",
    ]

    random.seed(args.seed)
    random.shuffle(prompts)

    base_model = build_model(tokenizer, args.window, args.model_path, None, device)
    adapter_model = build_model(tokenizer, args.window, args.model_path, args.adapter_path, device)

    if args.grid:
        adapt_ratios = [0.5, 0.7, 0.9, 0.95]
        repeats = [20, 50, 100]
        trials_list = [5, 10]
        for ar in adapt_ratios:
            for rep in repeats:
                for tr in trials_list:
                    cur_prompts = prompts[:tr]
                    pre_b, post_b, delta_b, pct_b = run_test(
                        base_model, tokenizer, cur_prompts, rep, ar, args.window, args.ttt_batch
                    )
                    pre_a, post_a, delta_a, pct_a = run_test(
                        adapter_model, tokenizer, cur_prompts, rep, ar, args.window, args.ttt_batch
                    )
                    print(f"=== grid adapt_ratio={ar} repeat={rep} trials={tr} ===")
                    print(f"Base:  pre={pre_b:.4f} post={post_b:.4f} delta={delta_b:.4f} ({pct_b:.3f}%)")
                    print(f"Adapt: pre={pre_a:.4f} post={post_a:.4f} delta={delta_a:.4f} ({pct_a:.3f}%)")
    else:
        cur_prompts = prompts[: args.trials]
        pre_b, post_b, delta_b, pct_b = run_test(
            base_model, tokenizer, cur_prompts, args.repeat, args.adapt_ratio, args.window, args.ttt_batch
        )
        pre_a, post_a, delta_a, pct_a = run_test(
            adapter_model, tokenizer, cur_prompts, args.repeat, args.adapt_ratio, args.window, args.ttt_batch
        )

        print("=== Base ===")
        print(f"pre_loss:  {pre_b:.4f}")
        print(f"post_loss: {post_b:.4f}")
        print(f"delta:     {delta_b:.4f}")
        print(f"delta %:   {pct_b:.3f}%")
        print("")
        print("=== Adapter ===")
        print(f"pre_loss:  {pre_a:.4f}")
        print(f"post_loss: {post_a:.4f}")
        print(f"delta:     {delta_a:.4f}")
        print(f"delta %:   {pct_a:.3f}%")


if __name__ == "__main__":
    main()
