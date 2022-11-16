#!/usr/bin/env python3

import json

import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    """The quantum circuit that you will differentiate!

    Args:
        params (list(float)): The parameters for gates in the circuit

    Returns:
        (numpy.array): An expectation value.
    """
    qml.RY(params[0], 0)
    qml.RX(params[1], 1)
    return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))


def my_parameter_shift_grad(params, shift):
    """Your homemade parameter-shift rule function.
    
    NOTE: you cannot use qml.grad within this function

    Args:
        params (list(float)): The parameters for gates in the circuit

    Returns:
        gradient (numpy.array): The gradient of the circuit with respect to the given parameters.
    """

    def shift_vector(vector, pos, shift):
        """This function creates the "shifted parameters" to feed in the circuit

        Args:
            vector (list(float)): The parameter vector to shift
            pos (int): The component to shift
            shift (float) : Value of the shift

        Return:
            shifted_vectors : (Tuple) Vectors of with opposite shifts
        """
        vector_plus  = np.copy(vector)
        vector_minus = np.copy(vector)

        vector_plus[pos]  += shift 
        vector_minus[pos] -= shift

        return (vector_plus, vector_minus)

    gradient = np.zeros_like(params)

    for i in range(len(params)):
        shift_plus, shift_minus = shift_vector(params, i, shift)
        gradient[i] = (circuit(shift_plus) - circuit(shift_minus) )

    gradient = gradient / (2*np.sin(shift))
    print('GRAD', gradient)
    return np.round_(gradient, decimals=5).tolist()
    

def run(test_case_input: str) -> str:
    params, shift = json.loads(test_case_input)
    gradient = my_parameter_shift_grad(params, shift)
    return str(gradient)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your gradient isn't quite right!"


test_cases = [['[[0.75, 1.0], 1.23]', '[-0.68164, -0.84147]']]

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
