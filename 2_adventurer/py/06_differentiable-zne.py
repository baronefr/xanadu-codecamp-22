import json
import pennylane as qml
import pennylane.numpy as np

dev_ideal = qml.device("default.mixed", wires=2)  # no noise
dev_noisy = qml.transforms.insert(qml.DepolarizingChannel, 0.05, position="all")(
    dev_ideal
)

def U(angle):
    """A quantum function containing one parameterized gate.

    Args:
        angle (float): The phase angle for an IsingXY operator
    """
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.CNOT(wires=[0, 1])
    qml.PauliZ(1)
    qml.IsingXY(angle, [0, 1])
    qml.S(1)

@qml.qnode(dev_noisy)
def circuit(angle):
    """A quantum circuit made from the quantum function U.

    Args:
        angle (float): The phase angle for an IsingXY operator
    """
    U(angle)
    return qml.state()

@qml.tape.stop_recording()
def circuit_ops(angle):
    """A function that outputs the operations within the quantum function U.

    Args:
        angle (float): The phase angle for an IsingXY operator
    """
    with qml.tape.QuantumTape() as tape:
        U(angle)
    return tape.operations

########################### our code ###############################

@qml.qnode(dev_noisy)
def global_fold_circuit(angle, n, s):
    """Performs the global circuit folding procedure.

    Args:
        angle (float): The phase angle for an IsingXY operator
        n: The number of times U^\dagger U is applied
        s: The integer defining L_s ... L_d.
    """
    assert s <= len(
        circuit_ops(angle)
    ), "The value of s is upper-bounded by the number of gates in the circuit."

    U(angle)  # Original circuit application

    # (U^\dagger U)^n
    for i in range(n):
        qml.adjoint(U)(angle)
        U(angle)

    subU = circuit_ops(angle)[(s-1)::] # selecting the operators L_i from s to d

    # L_d^\dagger ... L_s^\dagger
    for op in subU[::-1]:  qml.adjoint(op)
    
    # L_s ... L_d
    for op in subU:  qml.apply(op)

    return qml.state()




####################################################################

def fidelity(angle, n, s):
    fid = qml.math.fidelity(global_fold_circuit(angle, n, s), circuit(angle))
    return np.round_(fid, decimals=5)


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    angle, n, s = json.loads(test_case_input)
    fid = fidelity(angle, n, s)
    return str(fid)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your folded circuit isn't quite right!"


test_cases = [['[0.4, 2, 3]', '0.79209']]

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