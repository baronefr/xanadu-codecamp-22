#!/usr/bin/env python3

import json
import pennylane as qml
import pennylane.numpy as np

def circuit(circuit_param):
    qml.RY(circuit_param, wires=0)
    qml.Hadamard(wires=0)
    qml.T(wires=0)



########################### our code ###############################

def state_purity(angle, phase, circuit_param, noise_param):

    """
    This function returns the purity of the output state after adding noise
    to the given circuit().

    Args:
        angle (float): The angle theta that parametrizes the initial quantum state
        phase (float): The phase phi that parametrizes the initial quantum state
        circuit_param (float): The angle that paramterizes the RY rotation in circuit(alpha)
        noise_param (float): The angle that paramterizes the CRX gate in the circuit modelling the noise

    Returns:
        (float): Purity of the state after going through the noisy circuit
    """

    def noise(noise_param, wires):

        """Implements the circuit that models the noise added after each gate. Do not return anything."""
        # Apply CRX and CX
        print('aaaa', [k for k in wires[::-1]])
        qml.CRX(noise_param, wires= [k for k in wires[::-1]] )
        qml.CNOT(wires = wires)

    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def noisy_circuit(angle, phase, circuit_param, noise_param):

        """Implements transformed circuit with state preparation at the beginning, and noise inserted
        after each gate.

        Returns: Whatever is useful for you to calculate the purity!"""

        # Put your code here #
        # Don't forget to prepare the initial state
        # If you use a quantum transform to add noise, use it within this circuit
        # STATEPREP----RY(alpha)-----H-------T-----
        #                         |       |       |
        #                       NOISE   NOISE   NOISE

        # STATE PREPARATION:
        qml.RX(angle, wires = 0)
        qml.PhaseShift(- 1.5*np.pi + phase, wires = 0)
        #noise(noise_param, [0,1])

        # RY
        qml.RY(circuit_param, wires = 0)
        noise(noise_param, [1,0])
        
        # HADAMARD
        qml.Hadamard(wires = 0)
        noise(noise_param, [1,0])

        # T
        qml.T(wires = 0)
        noise(noise_param, [1,0])

        return qml.density_matrix(wires = 0)

        
    # Feel free to add here any code or helper functions, if you need them.
    rho = noisy_circuit(angle, phase, circuit_param, noise_param)
    rho_squared = rho @ rho
    print(np.trace(rho_squared))
    return np.trace(rho_squared) # Return the purity in terms of the calculated expectation values.


####################################################################






# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = state_purity(*ins)

    return str(np.real_if_close(output))

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
    ), "Your calculated purity isn't quite right."


test_cases = [['[0.1,0.2,0.3,0.4]', '0.9647'], ['[0.5,0.3,0.5,0.7]', '0.928356']]

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
