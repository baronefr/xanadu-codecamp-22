import json
import pennylane as qml
import pennylane.numpy as np

def hydrogen_hamiltonian(d):
    """Creates the H_2 Hamiltonian from a separation distance.

    Args:
        d (float): The distance between a hydrogen atom and the hydrogen molecule's centre of mass.

    Returns:
        H (qml.Hamiltonian): The H_2 Hamiltonian.
        qubits (int): The number of qubits needed to simulate the H_2 Hamiltonian.
    """

    symbols = symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -d, 0.0, 0.0, d])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

    return H, qubits

def ansatz_template(param, wires):
    """The unitaries used for creating an ansatz for subsequent VQE calculations.

    Args:
        param (np.array): A single differentiable parameter
        wires (list(int)): A list of wires that the unitaries are applied to.
    """
    qml.BasisState([1, 1, 0, 0], wires=wires)
    qml.DoubleExcitation(param, wires=wires)

def VQE(qnode):
    """Performs a VQE routine given a QNode.

    Args:
        qnode (qml.QNode):
            The ansatz that will be optimized in order to find the ground state
            of molecular hydrogen.

    Retuns:
        final_energy (float): The final energy from the VQE routine.
    """
    param = np.array(0.0, requires_grad=True)
    num_iters = 20
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    for _ in range(num_iters):
        param = opt.step(qnode, param)

    final_energy = qnode(param)

    return final_energy

def qnode_ansatzes(d, scale_factors):
    """Generates ideal and mitigated qnodes.

    Args:
        d (float): The distance between a hydrogen atom and the hydrogen molecule's centre of mass.
        scale_factors (list(int)): A list of scale factors used for ZNE.

    Returns:
       qnode_ideal (qml.QNode): The ideal QNode (no noise).
       qnodies_mitgated (list(qml.QNode)): A list of QNodes that are mitigated. len(qnodes_mitigated) = len(scale_factors).
    """
    H, qubits = hydrogen_hamiltonian(d)

    def cost_fn(param):
        ansatz_template(param, wires=range(qubits))
        return qml.expval(H)

    noise_gate = qml.DepolarizingChannel
    noise_strength = 0.05

    dev_ideal = qml.device("default.mixed", wires=qubits)
    dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

    qnode_ideal = qml.QNode(cost_fn, dev_ideal)
    qnode_noisy = qml.QNode(cost_fn, dev_noisy)
    
    qnodes_mitigated = [qml.transforms.fold_global(qnode_noisy, scale_factor) for scale_factor in scale_factors]

    return qnode_ideal, qnodes_mitigated

def extrapolation(d, scale_factors):
    """Performs ZNE to obtain a zero-noise estimate on the ground state energy of H_2.

    Args:
        d (float): The distance between a hydrogen atom and the hydrogen molecule's centre of mass.
        scale_factors (list(int)): A list of scale factors used for ZNE.

    Returns:
        ideal_energy (float): The ideal energy from a noise-less VQE routine.
        zne_energy (float): The zero-noise estimate of the ground state energy of H_2.

        These two energies are returned in that order within a numpy array.
    """

    qnode_ideal, qnodes_mitigated = qnode_ansatzes(d, scale_factors)

    ideal_energy = np.round_(VQE(qnode_ideal), decimals=6)
    mitigated_energies = [VQE(qnode) for qnode in qnodes_mitigated]

    # Put your code here #
    coeffs = np.polyfit(scale_factors, mitigated_energies, 2)
    zne_energy = coeffs[-1]

    return np.array([ideal_energy, zne_energy]).tolist()

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    d = json.loads(test_case_input)
    scale_factors = [1, 2, 3]
    energies = extrapolation(d, scale_factors)
    return str(energies)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-2
    ), "Your extrapolation isn't quite right!"

test_cases = [['0.6614', '[-1.13619, -0.41168]']]

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