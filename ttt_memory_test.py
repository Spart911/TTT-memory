import argparse
import random
import torch
from transformers import AutoTokenizer

from my_model import TransformerModel, MAX_LEN, MODEL_CONFIGS, MODEL_SIZE


def build_arg_parser():
    p = argparse.ArgumentParser(description="TTT memory test (pre/post loss).")
    p.add_argument("--model-path", type=str, default="transformer_ttt_personachat_hf.pth")
    p.add_argument("--tokenizer-path", type=str, default="./tokenizer_hf")
    p.add_argument("--window", type=int, default=MAX_LEN)
    p.add_argument("--ttt-batch", type=int, default=MAX_LEN)
    p.add_argument("--adapt-ratio", type=float, default=0.99)
    p.add_argument("--pattern", type=str, default="My name is {name}.")
    p.add_argument("--repeat", type=int, default=500)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if MODEL_SIZE not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_SIZE: {MODEL_SIZE}. Options: {list(MODEL_CONFIGS.keys())}")
    num_blocks, dim, num_heads = MODEL_CONFIGS[MODEL_SIZE]
    vocab_size = len(tokenizer)

    model = TransformerModel(
        vocab_size,
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        window_size=args.window,
        max_seq_len=MAX_LEN,
        tie_word_embeddings=False,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    random.seed(args.seed)

    prompts = [
        "I work as a software developer and write Python code daily.",
        "I live in a small apartment with a cat and two plants.",
        "I prefer coffee in the morning and tea in the вечер.",
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

    pre_losses = []
    post_losses = []
    for i in range(args.trials):
        pattern = prompts[i % len(prompts)]
        text = (" " + pattern).join([""] * args.repeat)
        tokens = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=MAX_LEN,
            return_tensors="pt",
        )["input_ids"].to(device)

        B, L = tokens.shape
        if L < 4:
            raise ValueError("Sequence too short for test.")

        adapt_end = max(1, int((L - 1) * args.adapt_ratio))
        adapt_end = min(adapt_end, L - 2)

        with torch.no_grad():
            fast_deltas = model._init_fast_deltas(B)
            pre_loss = model._post_ttt_loss(tokens, adapt_end, L - 1, fast_deltas)

        # Inner-loop TTT updates on adapt segment
        fast_deltas = model._init_fast_deltas(B)
        for i in range(0, adapt_end, args.ttt_batch):
            batch_end = min(i + args.ttt_batch, adapt_end)
            fast_deltas, _, _ = model._ttt_update(
                tokens, i, batch_end, args.window, fast_deltas
            )

        with torch.no_grad():
            post_loss = model._post_ttt_loss(tokens, adapt_end, L - 1, fast_deltas)

        pre_losses.append(pre_loss.item())
        post_losses.append(post_loss.item())

    pre_avg = sum(pre_losses) / len(pre_losses)
    post_avg = sum(post_losses) / len(post_losses)
    print(f"pre_loss avg:  {pre_avg:.4f}")
    print(f"post_loss avg: {post_avg:.4f}")
    print("TTT memory improved" if post_avg < pre_avg else "TTT memory did NOT improve")


if __name__ == "__main__":
    main()
