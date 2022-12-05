#!/usr/bin/env python3

import json
import pennylane as qml
import pennylane.numpy as np

# Create the device used to simulate the quantum circuit 
# and to construct QNodes.
dev = qml.device("default.mixed", wires=2)

@qml.qnode(dev)
def circuit():
    """An error-less quantum circuit.

    Returns:
        qml.state():
            The quantum state.
    """
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    
    return qml.state()
    
@qml.qnode(dev)
def bitflip_circuit(p):
    """A quantum circuit that contains two bitflip errors.
    It is the same circuit as the one above, but with bitflip errors.

    Args:
        p (float):
            The bitflip probability.

    Returns:
        qml.state():
            The quantum state.
    """
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    qml.BitFlip(p, wires=0)
    qml.BitFlip(p, wires=1)

    return qml.state()  
    
def fidelities(probs):
    fids = np.zeros(len(probs))
    # For each bitflip-probability value in probs (list(float))...
    for i, p in enumerate(probs):
        # Compute the fidelity between the output of the noisy circuit
        # and the ideal circuit
        fid = qml.math.fidelity(bitflip_circuit(p), circuit())
        fids[i] = fid
    return np.round_(fids, decimals=5).tolist()
    
    
def run(test_case_input: str) -> str:
    probs = json.loads(test_case_input)
    fids = fidelities(probs)
    return str(fids)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your circuit isn't quite right!"


test_cases = [['[0.05, 0.1, 0.15, 0.2, 0.25]', '[0.905, 0.82, 0.745, 0.68, 0.625]']]

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

