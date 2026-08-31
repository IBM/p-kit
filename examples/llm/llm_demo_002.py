"""
llm_demo_002.py

Train from scratch and compare:
  1. SparsePBitLM - our p-kit based LM
  2. Tiny Hugging Face GPT

NOTE: SparsePBitLM uses much fewer trainable parameters and no
backpropagation through the p-bit reservoir!!!

Both use the same character vocabulary, training corpus and test corpus.
Metrics: accuracy, perplexity, training time and generated text.

Requires: pip install "transformers[torch]"

Result:
    p-kit: 6,048 trainable weights
    GPT:   103,488 trainable parameters

    p-kit still reaches 54.9% accuracy vs 82.9% for GPT
    and generates partially structured text.
"""

import math, time
import numpy as np
import torch

#import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from p_kit.llm.llm_models import SparsePBitLM


TRAIN = """
the cat sat on the mat.
the dog sat by the door.
the cat looked at the dog.
the dog looked at the cat.
the cat walked to the door.
the dog walked to the mat.
the small robot saw the cat.
the small robot saw the dog.
the robot moved to the door.
the robot moved to the mat.
""" * 20

TEST = """
the cat moved to the door.
the dog moved to the mat.
the robot looked at the cat.
"""

VOCAB = sorted(set(TRAIN))
C2I = {c: i for i, c in enumerate(VOCAB)}


def encode(text):
    return [C2I[c] for c in text]


def evaluate_pkit(model, text):
    model.reset()
    correct = total = 0
    nll = 0.0

    for current, target in zip(text[:-1], text[1:]):
        p = model.next_char_probabilities(current, temperature=1.0)
        target_id = model.char_to_id[target]
        correct += int(np.argmax(p) == target_id)
        nll -= math.log(max(float(p[target_id]), 1e-12))
        total += 1

    ce = nll / total
    return correct / total, math.exp(ce)


def train_gpt(context=32, epochs=15):
    ids = encode(TRAIN)
    blocks = [
        torch.tensor(ids[i:i + context])
        for i in range(0, len(ids) - context, context // 2)
    ]

    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=context,
        n_ctx=context,
        n_embd=64,
        n_layer=2,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        bos_token_id=None,
        eos_token_id=None,
    ))

    loader = DataLoader(blocks, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    start = time.perf_counter()
    model.train()

    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            loss = model(input_ids=batch, labels=batch).loss
            loss.backward()
            opt.step()

    return model, time.perf_counter() - start


def evaluate_gpt(model, text, context=32):
    ids = encode(text)
    correct = total = 0
    nll = 0.0
    model.eval()

    with torch.no_grad():
        for t in range(1, len(ids)):
            x = torch.tensor(ids[max(0, t-context):t]).unsqueeze(0)
            p = torch.softmax(model(x).logits[0, -1], dim=-1)

            correct += int(torch.argmax(p) == ids[t])
            nll -= math.log(max(float(p[ids[t]]), 1e-12))
            total += 1

    ce = nll / total
    return correct / total, math.exp(ce)


def generate_gpt(model, prompt, length=100, temperature=0.4, context=32):
    out = prompt
    model.eval()

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor(encode(out[-context:])).unsqueeze(0)
            p = torch.softmax(model(x).logits[0, -1] / temperature, dim=-1)
            out += VOCAB[int(torch.multinomial(p, 1))]

    return out


def main():
    np.random.seed(7)
    torch.manual_seed(7)

    print("Training p-kit model...")
    pkit = SparsePBitLM(
        n_pbits=256, degree=8, input_fanout=16,
        recurrent_scale=0.40, input_scale=1.4,
        memory_scale=0.35, seed=7
    )

    t = time.perf_counter()
    pkit.fit(TRAIN, washout=30, ridge=0.05)
    pkit_time = time.perf_counter() - t
    pkit_acc, pkit_ppl = evaluate_pkit(pkit, TEST)

    print("Training Hugging Face GPT...")
    gpt, gpt_time = train_gpt()
    gpt_acc, gpt_ppl = evaluate_gpt(gpt, TEST)

    print("\nResults")
    print(f"{'':18} {'p-kit':>10} {'GPT':>10}")
    print(f"{'Accuracy':18} {pkit_acc:10.3f} {gpt_acc:10.3f}")
    print(f"{'Perplexity':18} {pkit_ppl:10.3f} {gpt_ppl:10.3f}")
    print(f"{'Training time (s)':18} {pkit_time:10.2f} {gpt_time:10.2f}")
    print(f"{'Trainable weights':18} {pkit.readout.size:10d} {sum(p.numel() for p in gpt.parameters()):10d}")

    prompt = "the robot "
    print("\np-kit:")
    print(pkit.generate(prompt, n_chars=100, temperature=0.15))

    print("\nGPT:")
    print(generate_gpt(gpt, prompt))


if __name__ == "__main__":
    main()