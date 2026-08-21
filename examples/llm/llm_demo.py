"""

As LLMs are a hot topic we try to implement one using p-kit.
Implementation is in SparsePBitLM. It is a research POC (proof of concept).

In the future SparsePBitLM could run on a RP2350 micro-controller
or be translated to quantum circuit/backend.

"""

from llm_model import SparsePBitLM


CORPUS = """
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


def main():
    
    model = SparsePBitLM(
        n_pbits=128,
        degree=6,
        input_fanout=12,
        beta=1.4,
        recurrent_scale=0.40,
        input_scale=1.4,
        seed=7,
    )

    stats = model.fit(
        CORPUS,
        sweeps=1,
        washout=30,
        ridge=0.05,
    )

    print("Sparse p-bit LM demo")
    print("--------------------")
    print(f"p-bits:       {model.n_pbits}")
    print(f"degree:       {model.degree}")
    print(f"edges:        {len(model.edge_w)}")
    print(f"vocabulary:   {stats.vocabulary_size}")
    print(f"train samples:{stats.n_samples}")
    print(f"train accuracy: {stats.accuracy:.3f}")
    print()

    for seed in ("the cat ", "the dog ", "the robot "):
        generated = model.generate(
            seed,
            n_chars=120,
            sweeps=1,
            temperature=0.15,
        )
        print(f"Seed: {seed!r}")
        print(generated)
        print()

    # The model contains a real p-kit circuit.
    # J and h can be accessed directly from model.circuit.
    J, h = model.dense_J_h("t")
    nonzero_J = (J != 0).sum()
    nonzero_h = (h != 0).sum()

    print("p-kit representation")
    print("--------------------")
    print(f"J shape:       {J.shape}")
    print(f"nonzero J:     {nonzero_J} / {J.size}")
    print(f"nonzero h:     {nonzero_h} / {h.size}")
    print(f"J sparsity:    {1.0 - nonzero_J / J.size:.4f}")

if __name__ == "__main__":
    main()
