from torch.utils.data import Dataset
from tqdm import tqdm
import random
import json

def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, add_prompt_suffix=None, add_system_prompt=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if add_prompt_suffix is not None:
            chat += add_prompt_suffix
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        if add_system_prompt:
            chat = [{"role": "system", "content": add_system_prompt}] + chat
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if add_prompt_suffix is not None:
            prompt += add_prompt_suffix
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDatasetWithLabel(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        add_prompt_suffix=None,
        use_golden_response=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        add_think_token = getattr(self.strategy.args, "add_think_token", 0)
        add_system_prompt = getattr(self.strategy.args, "add_system_prompt", None)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        indice = 0
        count = 0
        if use_golden_response:
            with open("code/data/all_deepscaler_dataset.json", "r") as f:
                all_dataset = json.load(f)
        for data in tqdm(dataset, desc="Preprocessing data"):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template, add_prompt_suffix, add_system_prompt)
            if apply_chat_template and add_think_token != 0:
                prompt = prompt + "<think>"
            if use_golden_response:
                prompt_raw = data[input_key]
                if prompt_raw not in all_dataset:
                    assert False
                self.prompts.append({"prompt": prompt, "label": data[label_key], "indice": indice, "golden_response": all_dataset.get(prompt_raw)})
            else:
                self.prompts.append({"prompt": prompt, "label": data[label_key], "indice": indice})
            indice += 1

        # breakpoint()
        for sample in random.sample(self.prompts, min(3, len(self.prompts))):
            print(sample)
            print("="*20)

    def get_all_prompts(self):
        return self.prompts

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
