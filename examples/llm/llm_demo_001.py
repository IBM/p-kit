"""
llm_demo_001.py

As LLMs are a hot topic we try to implement one using p-kit.
Implementation is in SparsePBitLM. It is a research POC.

The recurrent reservoir and ReLU nonlinear block are both
p-kit circuits executed by CaSuDaSolver. Only the final
ridge-regression readout is trained with NumPy.

In the future SparsePBitLM could run on an RP2350
microcontroller or be translated to a quantum circuit/backend.
"""

from p_kit.llm.llm_models import SparsePBitLM


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
        recurrent_scale=0.40,
        input_scale=1.4,
        memory_scale=0.35,
        seed=7,
    )

    print(
        "Training p-kit reservoir + ReLU..."
    )

    stats = model.fit(
        CORPUS,
        sweeps=1,
        washout=30,
        ridge=0.05,
    )

    print()
    print("p-bits:", model.n_pbits)

    print(
        "reservoir connections:",
        model.n_connections,
    )

    print(
        "reservoir sparsity:",
        f"{100 * model.sparsity:.1f}%",
    )

    print(
        "training accuracy:",
        f"{stats.accuracy:.3f}",
    )

    print("\nGenerated text:")

    for seed in (
        "the cat ",
        "the dog ",
        "the robot ",
    ):

        print(
            model.generate(
                seed,
                n_chars=120,
                temperature=0.15,
            )
        )

        print()

if __name__ == "__main__":
    main()