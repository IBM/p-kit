from .decorators import *


@pcircuit(n_pbits=3)
class ANDGate:
    """
    Probabilistic implementation of an AND gate

    Attributes:
        input1 (Port): First input port
        input2 (Port): Second input port
        output (Port): Output port
    """

    input1 = Port("input1")
    input2 = Port("input2")
    output = Port("output")

    J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])
    h = np.array([[1], [1], [-2]])


@pcircuit(n_pbits=3)
class ORGate:
    """
    Probabilistic implementation of an OR gate

    Attributes:
        input1 (Port): First input port
        input2 (Port): Second input port
        output (Port): Output port
    """

    input1 = Port("input1")
    input2 = Port("input2")
    output = Port("output")

    J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])
    h = np.array([[-1], [-1], [2]])


@pcircuit(n_pbits=5)
class FullAdder:
    """
    Probabilistic implementation of a Full Adder

    Attributes:
        input1 (Port): First input port
        input2 (Port): Second input port
        carryin (Port): Carry in port
        sumout (Port): Sum output port
        carryout (Port): Carry out port
    """

    input1 = Port("input1")
    input2 = Port("input2")
    carryin = Port("carryin")
    sumout = Port("sumout")
    carryout = Port("carryout")

    J = np.array(
        [
            [0, -1, -1, 1, 2],
            [-1, 0, -1, 1, 2],
            [-1, -1, 0, 1, 2],
            [-1, -1, 1, 0, -2],
            [2, 2, 2, -2, 0],
        ]
    )

    h = np.array([[-1], [-1], [-1], [1], [2]])
