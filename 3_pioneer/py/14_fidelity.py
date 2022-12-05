import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy


@qml.qfunc_transform
def rotate_rots(tape, params):
    for op in tape.operations + tape.measurements:
        if op.name == "RX":
            if list(op.wires) == [0]:
                qml.RX(op.parameters[0] + params[0], wires=op.wires)
            else:
                qml.RX(op.parameters[0] + params[1], wires=op.wires)
        elif op.name == "RY":
            if list(op.wires) == [0]:
                qml.RY(op.parameters[0] + params[2], wires=op.wires)
            else:
                qml.RY(op.parameters[0] + params[3], wires=op.wires)
        elif op.name == "RZ":
            if list(op.wires) == [0]:
                qml.RZ(op.parameters[0] + params[4], wires=op.wires)
            else:
                qml.RZ(op.parameters[0] + params[5], wires=op.wires)
        else:
            qml.apply(op)

def circuit():
    qml.RX(np.pi / 2, wires=0)
    qml.RY(np.pi / 2, wires=0)
    qml.RZ(np.pi / 2, wires=0)
    qml.RX(np.pi / 2, wires=1)
    qml.RY(np.pi / 2, wires=1)
    qml.RZ(np.pi / 2, wires=1)



def optimal_fidelity(target_params, pauli_word):

    """This function returns the maximum fidelity between the final state that we obtain with only
    Pauli rotations with respect to the state we obtain with the target circuit

    Args:
        - target_params (list(float)): List of the two parameters in the target circuit. The first is
        the parameter of the Pauli Rotation, the second is the parameter of the CRX gate.
        - pauli_word: A string that is either 'X', 'Y', or 'Z', depending on the Pauli rotation
        implemented by the target circuit.
    Returns:
        - (float): Maximum fidelity between the states produced by both circuits.
    """

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def target_circuit(target_params, pauli_word):
        """This QNode is target circuit whose effect we want to emulate"""
        # Put your code here #
        qml.PauliRot(target_params[0], pauli_word,  wires=0)
        qml.CRX(target_params[1], wires = [0,1])
        qml.T(wires = 0)
        qml.S(wires = 1)
        
        return qml.state()

    @qml.qnode(dev)
    def rotated_circuit(rot_params):
        """This QNode is the available circuit, with rotated parameters

        Inputs:
        rot_params list(float): A list containing the values of the independent rotation parameters
        for each gate in the available circuit. The order will not matter, since you are optimizing
        for these and will return the minimal value of a cost function (related
        to the fidelity)
        """
        rotate_rots(rot_params)(circuit)()
        
        return qml.state()

    # Write an optimization routine for an adequate cost function.
    def cost(rot_params):
        return qml.math.fidelity(rotated_circuit(rot_params),target_circuit(target_params, pauli_word) )
    
    epochs = 1000
    lr = .1

    grad = qml.grad(cost)
    rot_params = np.random.rand(6) * np.pi
    
    for epoch in range(epochs):
        rot_params += lr * grad(rot_params)
        
    return cost(rot_params)



def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = optimal_fidelity(*ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    """
    Compare solution with expected.

    Args:
            solution_output: The output from an evaluated solution. Will be
            the same type as returned.
            expected_output: The correct result for the test case.

    Raises:
            ``AssertionError`` if the solution output is incorrect in any way.
    """

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-2
    ), "Your calculated optimal fidelity isn't quite right."



test_cases = [['[[1.6,0.9],"X"]', '0.9502'], ['[[0.4,0.5],"Y"]', '0.9977']]


for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")