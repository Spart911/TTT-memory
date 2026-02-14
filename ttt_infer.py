import argparse
import torch
from transformers import AutoTokenizer

from my_model import TransformerModel, MAX_LEN, MODEL_CONFIGS, MODEL_SIZE
from ttt_decode import TTTDecoder


def build_arg_parser():
    p = argparse.ArgumentParser(description="TTT inference chat with KV-cache decoding.")
    p.add_argument("--model-path", type=str, default="transformer_ttt_personachat_hf.pth")
    p.add_argument("--tokenizer-path", type=str, default="./tokenizer_hf")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--window", type=int, default=MAX_LEN)
    p.add_argument("--ttt-batch", type=int, default=MAX_LEN)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--min-tokens-before-ttt-update", type=int, default=0)
    p.add_argument("--no-flush", action="store_true")
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
    # Drop non-persistent buffers if present in checkpoint
    state.pop("blocks.0.attn.mask_cache", None)
    state.pop("blocks.1.attn.mask_cache", None)
    state.pop("blocks.2.attn.mask_cache", None)
    state.pop("blocks.3.attn.mask_cache", None)
    model.load_state_dict(state, strict=False)
    model.eval()

    decoder = TTTDecoder(
        model,
        window_size=args.window,
        ttt_batch_size=args.ttt_batch,
    )

    history = []
    while True:
        if args.prompt is not None and not history:
            prompt = args.prompt
        else:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            break

        history.append(f"User: {prompt}")
        if len(history) > 10:
            history = history[-10:]

        full_prompt = "\n".join(history + ["Assistant:"])

        tokens = tokenizer(
            full_prompt,
            truncation=True,
            padding=False,
            max_length=MAX_LEN,
            return_tensors="pt",
        )["input_ids"].to(device)

        out_ids = decoder.generate(
            tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
            min_tokens_before_ttt_update=args.min_tokens_before_ttt_update,
            flush_ttt_update=not args.no_flush,
        )
        # Decode only newly generated tokens (avoid echoing full history)
        gen_ids = out_ids[0, tokens.size(1) :]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"Assistant: {answer}")

        history.append(f"Assistant: {answer}")
        if len(history) > 10:
            history = history[-10:]


if __name__ == "__main__":
    main()
