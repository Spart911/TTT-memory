import torch


class TTTDecoder:
    """Decode с mini-batch TTT-обновлениями и KV-кэшем."""

    def __init__(self, model, window_size=16, ttt_batch_size=4):
        self.model = model
        self.window_size = window_size
        self.ttt_batch_size = ttt_batch_size
        assert self.window_size >= self.ttt_batch_size, "WINDOW must be >= TTT_BATCH (k >= b)."

    def _init_fast_params(self):
        fast_params_per_block = []
        for blk in self.model.blocks:
            if blk.ttt_block:
                fast_params_per_block.append(blk.mlp_ttt.fast_params())
            else:
                fast_params_per_block.append(None)
        return fast_params_per_block

    def _sample(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / max(temperature, 1e-8)
        if top_k and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = cumprobs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -1e10), sorted_logits)
            logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(
        self,
        tokens,
        max_new_tokens=32,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        greedy=False,
        min_tokens_before_ttt_update=0,
        flush_ttt_update=True,
    ):
        tokens = tokens.to(self.model.embedding.embed.weight.device)
        B, L = tokens.shape
        assert B == 1, "Сейчас generate реализован для batch_size=1"

        fast_params_per_block = self._init_fast_params()
        logits, cache = self.model.prefill_kv_cache(tokens, fast_params_per_block)

        generated = tokens.clone()
        pending_start = L - 1 + max(min_tokens_before_ttt_update, 0)

        for _ in range(max_new_tokens):
            if greedy or temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = self._sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            generated = torch.cat([generated, next_token], dim=1)
            logits, cache = self.model.forward_step(
                generated[:, -1:], cache, fast_params_per_block
            )

            if generated.size(1) - 1 - pending_start >= self.ttt_batch_size:
                pos_start = pending_start
                pos_end = pending_start + self.ttt_batch_size
                fast_params_per_block, _, _ = self.model._ttt_update(
                    generated, pos_start, pos_end, self.window_size, fast_params_per_block
                )
                pending_start = pos_end

        if flush_ttt_update:
            final_end = generated.size(1) - 1
            if final_end > pending_start:
                fast_params_per_block, _, _ = self.model._ttt_update(
                    generated, pending_start, final_end, self.window_size, fast_params_per_block
                )

        return generated
