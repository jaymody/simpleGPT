# SimpleGPT

Simple implementation of a GPT (training and inference) in PyTorch.

Basically my version of [nanoGPT](https://github.com/karpathy/nanoGPT) with some minor differences:

* Using [lightning](https://lightning.ai) to handle training.
* Using [einops](https://github.com/arogozhnikov/einops) for readable neural net code.
* Using [pydantic](https://docs.pydantic.dev/latest/) instead of [Poor Man's Configurator](https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/configurator.py#L2).
* Is even simpler (imo).

## Install Dependencies
```shell
pip install .
```

If you're developing changes to the codebase, use:
```shell
pip install -e ".[dev]"
```

## Run GPT2 Inference

Running inference on pre-trained GPT2 model:

```shell
python -m simplegpt.inference \
    "Alan Turing theorized that computers would one day become" \
    --model_name_or_ckpt_path "gpt2"
```

Pre-trained GPT2 models come in the sizes `gpt2`, `gpt2-medium`, `gpt2-large`,
and `gpt2-xl`.

Here's an example with more options:
```shell
python -m simplegpt.inference \
    "Alan Turing theorized that computers would one day become" \
    --model_name_or_ckpt_path "gpt2"
    --n_tokens_to_generate 40 \
    --batch_size 4 \
    --seed 321 \
    --temperature 0.2
```

## Train a Model from Scratch

Let's train a baby GPT model on the [tiny shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

First let's download the dataset text:

```ShellSession
$ mkdir data

$ cd data

$ wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

$ head -n 20 input.txt
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
```

Then, we'll split the dataset into a train and validation set by simply splitting the file at the 90% mark by line number:

```ShellSession
$ wc -l input.txt # count number of lines
40000

$ split -l 36000 input.txt  # 40000 * 0.9 = 36000

$ ls
input.txt
xaa
xab

$ wc -l xaa
3600

$ wc -l xab
4000

$ mv xaa train.txt

$ mv xab val.txt
```

Then, we'll need to encode the text into tokens:

```ShellSession
$ python -m simplegpt.data --text_file "data/train.txt" --output_file "data/train-char.bin" --tokenizer_name "char"
100%|████████████████████████████████████████████████████████████████████████████| 36000/36000 [00:00<00:00, 694079.64it/s]
number of tokens = 1016242
saving to data/train.bin

$ python -m simplegpt.data --text_file "data/val.txt" --output_file "data/val-char.bin" --tokenizer_name "char"
100%|██████████████████████████████████████████████████████████████████████████████| 4000/4000 [00:00<00:00, 719403.80it/s]
number of tokens = 99152
saving to data/val.bin
```

Here, we're using a character-level tokenizer which just tokenizes the text based on it's ascii value (which works on this dataset since it contains only ascii characters). We could have also used the regular BPE based tokenizer by instead passing in `--tokenizer_name "gpt2"`.

Finally, we train our model using the provided configuration file `configs/shakespeare_char.toml`:
```shell
python -m simplegpt.train configs/shakespeare_char.toml
```

If you used the regular "gpt2" tokenizer instead of the character-level tokenizer, use the `config/shakespeare.toml` config file:

On my 3090, this takes about 5 minutes to train. We can run inference on the
newly trained model by passing in the checkpoint path to `simplegpt.inference`:

```shell
python -m simplegpt.inference \
    "The lady doth protest" \
    --model_name_or_ckpt_path "models/tiny_shakespeare_char/ckpts/last.ckpt" \
    --n_tokens_to_generate 100
```

For the character-level model, this gives an output of:

```
==== Result 1 ====
 that I have seen.

QUEEN ELIZABETH:
The linealness hath been done to have it so.

KING RICHARD III:
And shall I live, if to hear a thousand affects
Are to be so still a son of the king.

QUEEN ELIZAB

==== Result 2 ====
; and with the time shall
With some men of the world and look for his head.

QUEEN MARGARET:
And leave me so, and leave the world alone.

KING HENRY VI:
What said Clarence to my son? what say you?
```

For the regularly tokenized model, this gives an output of:

```
==== Result 1 ====
; and so still so much,
That, were I madmen,--

PAULINA:
That's enough.
I must be so far gone, sir, sit by my side,
And leave it as you to part your company:
Good night.

LEONTES:
Thou'rt i' the or bad;
I have forgot already made thy beauty.

PAULINA:
A most unworthy and unnatural lord
Can do no

==== Result 2 ====
; and so I am:
I think there is not half the night in her
Until the fair ladies of York's lap,
And in my state and honour beauteous inn,
Why should I think be so deep a maidr'd her sweetor?

KING RICHARD III:
Madam, so I am a subject.

QUEEN ELIZABETH:
And shall I woo her?
```


## Todos
- [ ] Add support for fine-tuning.
- [ ] Actually reproduce GPT-2 (while I don't have the compute resources for this, I can at least run the model for a couple days on my 3090 and check that the train/val loss is what it's suppose to be).
- [ ] Add support for resuming training.
- [ ] Add support for top-p and top-k sampling.
- [ ] Use flash attention.
- [ ] Add support for lower precision training training.
