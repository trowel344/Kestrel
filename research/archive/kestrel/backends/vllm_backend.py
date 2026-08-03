import torch
from vllm import LLM, SamplingParams


class VLLMBackend:
    def __init__(
        self,
        model_name: str,
        draft_model_name: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "auto",
        speculative_config: dict | None = None,
    ):
        self.model_name = model_name
        self.draft_model_name = draft_model_name
        self.speculative_config = speculative_config or {}

        kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )

        if draft_model_name and speculative_config:
            kwargs["speculative_config"] = speculative_config

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    @property
    def has_draft_model(self) -> bool:
        return self.draft_model_name is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[str, dict]:
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate([prompt], params)
        result = outputs[0]
        text = result.outputs[0].text
        info = {
            "prompt_tokens": len(result.prompt_token_ids),
            "output_tokens": len(result.outputs[0].token_ids),
            "finish_reason": result.outputs[0].finish_reason,
        }
        return text, info

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        for output in self.llm.generate([prompt], params):
            yield output.outputs[0].text

    def shutdown(self):
        del self.llm
        torch.cuda.empty_cache()
