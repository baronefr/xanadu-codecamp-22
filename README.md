# PennyLane Codecamp 2022

```markdown
### Team: Pizza Kebab
* RANK: 7/429 teams
* SCORE: 4000pt
* TIME: 141351 mins
```

**Members**: Comuz ([Saverio Monaco](https://github.com/SaverioMonaco/)), Linuz ([Francesco Barone](https://github.com/baronefr))


<p align="center">
<img src='img/badges.png' width="40%">
</p>

<br><br>

## Challenges

### Explorer
* 1 ***[Differentiable ZNE: Noisy Circuits](1_explorer/01_differentiable-zne.ipynb)***
  Build a quantum circuit with noise.

* 2 ***[Fourier spectrum of quantum modelss: Build a model](1_explorer/02_fourier-spectrum.ipynb)***
  We introduce the basics of building quantum models with classical input data.

* 3 ***[Gradients: The parameter-shift rule](1_explorer/03_parameter_shift.ipynb)***
  Implement your own simple parameter-shift rule for the Pauli rotation gates.

* 4 ***[Quantum transforms and noise: Energy dissipation](1_explorer/04_energy-dissipation.ipynb)***
  We use quantum transforms to add energy dissipation to every gate in a single-wire circuit. We study the effects of energy dissipation in the purity of the final states.

* 5 ***[Universality: Working with one qubit](1_explorer/05_universality.ipynb)***
  In this challenge we will work with the definition of universality in one qubit

### Adventurer
* 6 ***[Differentiable ZNE: Global circuit folding](2_adventurer/06_differentiable-zne.ipynb)***
  Implement a global circuit folding routine

* 7 ***[Fourier spectrum of quantum models - Distance in Fourier space](2_adventurer/07_fourier-spectrum.ipynb)***
  In this challenge, we write a script that gives us a notion of how well a quantum model fits a given function.

* 8 ***[Gradients: A four-term parameter-shift rule](2_adventurer/08_parameter-shift-CRX.ipynb)***
  Implement your own parameter-shift rule for the controlled X/Y/Z gates

* 9 ***[Quantum transforms and noise: A faulty superconducting device](2_adventurer/09_faulty-supercomputer.ipynb)***
  We use quantum transforms to model a faulty real-life superconducting device.

* 10 ***[Universality: U3 and CNOT decomposition](2_adventurer/10_universality.ipynb)***
  This time there will be more qubits. In this challenge we will try to define a given matrix with only U3 and CNOT gates.

### Pioneer
* 11 **[Differentiable ZNE: Error-mitigated VQE](3_pioneer/11_vqe.ipynb)**
    Use ZNE to obtain a zero-noise estimate on the Hydrogen molecule ground state energy.

* 12 **[Fourier spectrum of quantum models - Coefficients and frequencies](3_pioneer/12_fourier.ipynb)**
  We build a quantum model and use the Fourier module to extract its spectrum and Fourier coefficients.

* 13 **[Gradients: Adjoint differentiation](3_pioneer/13_adjoint.ipynb)**
  Implement a seemingly harmless adjoint differentiation protocol!

* 14 **[Quantum transforms and noise: Maximizing fidelity](3_pioneer/14_fidelity.ipynb)**
  We pioneer on the use of transforms to see how to best emulate a quantum circuit using a different one.

* 15 **[Universality: H, T and CNOT decomposition](3_pioneer/15_universality.ipynb)**
  Making use of different equivalences, you will be asked to construct a particular operator simply using H, T and CNOT...



<br><br>

#### How to Run:
1. Create a pip environment called `codecamp`:
``python3 -m venv codecamp``
2. Activate the environment:
``source ./codecamp/bin/activate``
3. Install the required packages from `./requirements.txt`:
``python3 -m pip install -r ./requirements.txt``
