import numpy as np

from qiskit_machine_learning.utils.loss_functions import Loss, CrossEntropyLoss

from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, Minimizer

from qiskit_machine_learning.utils import  derive_num_qubits_feature_map_ansatz
from typing import Callable
from qiskit_machine_learning.algorithms.classifiers import VQC
from utility.froze import FrozenQNN   


class ExtendedVQC(VQC):
    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | None = None,
        ansatz: QuantumCircuit | None = None,
        loss: str | Loss = "cross_entropy",
        optimizer: Optimizer | Minimizer | None = None,
        warm_start: bool = False,
        initial_point: np.ndarray | None = None,
        callback: Callable[[np.ndarray, float], None] | None = None,
        *,
        sampler: BaseSampler | None = None,
    ) -> None:

        num_qubits, feature_map, ansatz = derive_num_qubits_feature_map_ansatz(
            num_qubits, feature_map, ansatz
        )

        # construct circuit
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._num_qubits = num_qubits
        self._circuit = QuantumCircuit(self._num_qubits)
        self._circuit.compose(self.feature_map, inplace=True)
        self._circuit.compose(self.ansatz, inplace=True)

        neural_network = FrozenQNN(
            sampler=sampler,
            circuit=self._circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=self._get_interpret(2),
            output_shape=2,
            input_gradients=False,
        )

