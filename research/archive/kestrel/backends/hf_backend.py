import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFBackend:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_model_len: int = 2048,
    ):
        self.model_name = model_name
        self.device = device

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}
        torch_dtype = dtype_map.get(dtype, "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.max_model_len = max_model_len

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[str, dict]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_len:].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        info = {
            "prompt_tokens": prompt_len,
            "output_tokens": len(generated_ids),
            "finish_reason": "length" if len(generated_ids) >= max_tokens else "stop",
        }
        return text, info

    def shutdown(self):
        del self.model
        torch.cuda.empty_cache()
