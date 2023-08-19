import fire
import torch

from .model import GPT
from .utils import get_device, get_tokenizer, set_seed


def from_openai(model_name):
    model = GPT.from_pretrained(model_name)
    tokenizer = get_tokenizer("gpt2")
    return model, tokenizer


def from_ckpt_path(ckpt_path):
    ckpt = torch.load(ckpt_path)
    hparams = ckpt["hyper_parameters"]
    tokenizer = get_tokenizer(hparams["tokenizer_name"])
    model = GPT(**hparams["model"])
    state_dict = {k.removeprefix("gpt."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model, tokenizer


def main(
    prompt,
    n_tokens_to_generate=30,
    batch_size=4,
    temperature=0.25,
    model_name_or_ckpt_path="gpt2",
    seed=1234,
):
    set_seed(seed)
    device = get_device()

    if model_name_or_ckpt_path in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model, tokenizer = from_openai(model_name_or_ckpt_path)
    else:
        model, tokenizer = from_ckpt_path(model_name_or_ckpt_path)
    model = model.to(device)
    model.eval()

    input_ids = torch.tensor(tokenizer.encode(prompt)).repeat(batch_size, 1).to(device)

    output_ids = model.generate(input_ids, n_tokens_to_generate, temperature)
    for i, ids in enumerate(output_ids):
        print(f"==== Result {i + 1} ====")
        print(tokenizer.decode(list(ids)))
        print()


if __name__ == "__main__":
    fire.Fire(main)
