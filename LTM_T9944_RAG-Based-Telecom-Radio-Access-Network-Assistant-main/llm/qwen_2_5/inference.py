import torch

from llm.qwen_2_5.tokenizer_loader import (
    QwenTokenizerLoader
)

from llm.qwen_2_5.model_loader import (
    QwenModelLoader
)


class QwenInference:

    def __init__(self):

        self.tokenizer_loader = (
            QwenTokenizerLoader()
        )

        self.model_loader = (
            QwenModelLoader()
        )

        self.tokenizer = (
            self.tokenizer_loader.load()
        )

        self.model = (
            self.model_loader.load_model()
        )

    def generate(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.7
    ):

        if prompt is None:
            raise ValueError(
                "Prompt cannot be None."
            )

        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Telecom RAN AI Assistant. "
                    "Answer questions about 4G, 5G, "
                    "O-RAN, RAN KPIs, network performance, "
                    "and 3GPP specifications accurately."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                )
            )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = outputs[
            0
        ][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()