import torch
from dataclasses import dataclass


@dataclass
class DraftResult:
    tokens: list[int]
    confidence: float
    block_size: int
    draft_logits: list[torch.Tensor] | None = None
    verify_logits: torch.Tensor | None = None


@dataclass
class AcceptResult:
    accepted: list[int]
    n_accepted: int
    n_draft: int
    done: bool


class DFlashDraftEngine:
    def __init__(self, engine, draft_k: int = 2, verify_k: int = 8, block_size: int = 8):
        self.engine = engine
        self.draft_k = draft_k
        self.verify_k = verify_k
        self.block_size = block_size
        self._loaded = True

    def load(self):
        pass

    @torch.inference_mode()
    def draft(self, anchor_token: int, position_id: int | None = None) -> DraftResult:
        engine = self.engine
        device = engine.device
        tokens = []
        draft_logits = []
        cur_token = torch.tensor([[anchor_token]], device=device)

        for i in range(self.block_size):
            pos = None
            if position_id is not None:
                pos = torch.tensor([[position_id + i]], device=device)
            logits, _ = engine.forward(input_ids=cur_token, position_ids=pos, top_k=self.draft_k)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            draft_logits.append(logits[:, -1, :].cpu())
            cur_token = next_token

        avg_conf = sum(l.softmax(-1).max().item() for l in draft_logits) / len(draft_logits)
        return DraftResult(tokens=tokens, confidence=avg_conf, block_size=self.block_size, draft_logits=draft_logits)

    @torch.inference_mode()
    def verify(self, draft_tokens: list[int], position_id: int | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        engine = self.engine
        device = engine.device
        n = len(draft_tokens)
        all_tokens = torch.tensor([draft_tokens], device=device)
        pos = None
        if position_id is not None:
            pos = torch.arange(position_id, position_id + n, device=device).unsqueeze(0)
        logits, _ = engine.forward(input_ids=all_tokens, position_ids=pos, top_k=self.verify_k)
        draft_logits_list = [logits[:, j, :].cpu() for j in range(n)]
        return draft_logits_list, logits

    def accept(self, draft_result: DraftResult, verify_logits: torch.Tensor | None = None) -> AcceptResult:
        draft_toks = draft_result.tokens
        n = len(draft_toks)
        if verify_logits is None:
            return AcceptResult(accepted=draft_toks, n_accepted=n, n_draft=n, done=True)

        accepted = []
        for i in range(n):
            draft_id = draft_toks[i]
            verify_id = verify_logits[:, i, :].argmax(dim=-1).item()
            if draft_id == verify_id:
                accepted.append(draft_id)
            else:
                break

        n_acc = len(accepted)
        return AcceptResult(
            accepted=accepted,
            n_accepted=n_acc,
            n_draft=n,
            done=n_acc == n,
        )

    @torch.inference_mode()
    def speculative_generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> tuple[str, dict]:
        engine = self.engine
        tokenizer = engine.tokenizer
        device = engine.device

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        seq_len = prompt_len

        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        logits, _ = engine.forward(input_ids=input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]
        seq_len += 1

        stats = {"draft_runs": 0, "verify_runs": 0, "draft_tokens": 0, "accepted_tokens": 0}

        while len(generated) < max_new_tokens:
            pos = seq_len - 1
            draft_res = self.draft(anchor_token=generated[-1], position_id=pos)
            stats["draft_runs"] += 1
            stats["draft_tokens"] += len(draft_res.tokens)

            if not draft_res.tokens:
                break

            verify_logits_list, verify_logits = self.verify(
                draft_tokens=draft_res.tokens, position_id=pos + 1
            )
            stats["verify_runs"] += 1

            acc = self.accept(draft_res, verify_logits)
            stats["accepted_tokens"] += acc.n_accepted
            generated.extend(acc.accepted)
            seq_len += acc.n_accepted

            if not acc.done:
                fallback_token = verify_logits[:, acc.n_accepted, :].argmax(dim=-1).item()
                generated.append(fallback_token)
                seq_len += 1

            if len(generated) >= max_new_tokens:
                break

        t1.record()
        torch.cuda.synchronize()
        total_ms = t0.elapsed_time(t1)
        text = tokenizer.decode(generated[:max_new_tokens], skip_special_tokens=True)

        info = {
            "total_time_ms": total_ms,
            "new_tokens": min(len(generated), max_new_tokens),
            "tok_per_sec": min(len(generated), max_new_tokens) / (total_ms / 1000),
            "draft_runs": stats["draft_runs"],
            "verify_runs": stats["verify_runs"],
            "draft_tokens": stats["draft_tokens"],
            "accepted_tokens": stats["accepted_tokens"],
            "acceptance_rate": stats["accepted_tokens"] / max(stats["draft_tokens"], 1),
            "prompt_len": prompt_len,
        }
        return text, info

    @property
    def is_loaded(self) -> bool:
        return self._loaded
