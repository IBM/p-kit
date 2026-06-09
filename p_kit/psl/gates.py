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


@pcircuit(n_pbits=4)
class XORGate:
    """
    Probabilistic implementation of an XOR gate using auxiliary bit

    Order: [input1, input2, output, aux]
    where aux = input1 AND input2

    Attributes:
        input1 (Port): First input port
        input2 (Port): Second input port
        output (Port): XOR output port
        aux (Port): Auxiliary bit (internal AND)
    """

    input1 = Port("input1")
    input2 = Port("input2")
    output = Port("output")
    aux = Port("aux")

    J = np.array([
        [0, -3, 2, 6],
        [-3, 0, 2, 6],
        [2, 2, 0, -4],
        [6, 6, -4, 0],
    ])
    h = np.array([[3], [3], [-2], [-6]])


@pcircuit(n_pbits=4)
class XNORGate:
    """
    Probabilistic implementation of an XNOR gate using auxiliary bit
    XNOR = NOT(XOR), outputs 1 when inputs match

    Order: [input1, input2, output, aux]
    where aux = input1 AND input2

    Attributes:
        input1 (Port): First input port
        input2 (Port): Second input port
        output (Port): XNOR output port
        aux (Port): Auxiliary bit (internal AND)
    """

    input1 = Port("input1")
    input2 = Port("input2")
    output = Port("output")
    aux = Port("aux")

    J = np.array([
        [0, -3, -2, 6],
        [-3, 0, -2, 6],
        [-2, -2, 0, 4],
        [6, 6, 4, 0],
    ])
    h = np.array([[3], [3], [2], [-6]])


@pcircuit(n_pbits=4)
class HalfAdder:
    """
    Probabilistic implementation of a Half Adder
    Sum = A XOR B, Carry = A AND B

    Order: [input1, input2, sumout, carryout]

    Attributes:
        input1 (Port): First input port
        input2 (Port): Second input port
        sumout (Port): Sum output port (XOR)
        carryout (Port): Carry output port (AND)
    """

    input1 = Port("input1")
    input2 = Port("input2")
    sumout = Port("sumout")
    carryout = Port("carryout")

    J = np.array([
        [0, -3, 2, 6],
        [-3, 0, 2, 6],
        [2, 2, 0, -4],
        [6, 6, -4, 0],
    ])
    h = np.array([[3], [3], [-2], [-6]])
