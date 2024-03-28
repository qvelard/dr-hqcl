from __future__ import annotations
from typing import Callable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, Minimizer

from ...neural_networks import SamplerQNN
from ...utils import derive_num_qubits_feature_map_ansatz
from ...utils.loss_functions import Loss
from .neural_network_classifier import NeuralNetworkClassifier

import torch
import torch.nn as nn

class CrossDistilledLoss(nn.Module):
    def __init__(self, num_old_classes, temperature=2.0):
        super(CrossDistilledLoss, self).__init__()
        self.num_old_classes = num_old_classes
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_new, logits_old, targets):
        # Separate logits for old and new classes
        logits_old = logits_old[:, :self.num_old_classes]
        logits_new = logits_new[:, self.num_old_classes:]

        # Compute cross-entropy loss for new classes
        loss_ce = self.criterion(logits_new, targets - self.num_old_classes)

        # Compute distillation loss for old classes
        probs_old = nn.functional.softmax(logits_old / self.temperature, dim=1)
        probs_new = nn.functional.softmax(logits_new / self.temperature, dim=1)
         #kl_div function is the Kullback-Leibler divergence which measures the difference between two probability distriutions 
        loss_distill = nn.functional.kl_div(nn.functional.log_softmax(logits_old / self.temperature, dim=1), probs_new, reduction='batchmean')
        

        # Combine losses
        loss = loss_ce + loss_distill

        return loss
    #un fonction qui appartient à une classe est une méthode, à partir des attributs (et pas de variables) de la classe on peut def des instances(occurences)
    #attribut de la classe différent de l'attribut de l'instance

class ExtendedVQC(NeuralNetworkClassifier):
     r"""A convenient Variational Quantum Classifier implementation.

    The variational quantum classifier (VQC) is a variational algorithm where the measured
    bitstrings are interpreted as the output of a classifier.

    Constructs a quantum circuit and corresponding neural network, then uses it to instantiate a
    neural network classifier.

    Labels can be passed in various formats, they can be plain labels, a one dimensional numpy
    array that contains integer labels like `[0, 1, 2, ...]`, or a numpy array with categorical
    string labels. One hot encoded labels are also supported. Internally, labels are transformed
    to one hot encoding and the classifier is always trained on one hot labels.

    Multi-label classification is not supported. E.g., :math:`[[1, 1, 0], [0, 1, 1], [1, 0, 1]]`.
    """

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
        """
        Args:
            num_qubits: The number of qubits for the underlying QNN.
                If ``None`` is given, the number of qubits is derived from the
                feature map or ansatz. If neither of those is given, raises an exception.
                The number of qubits in the feature map and ansatz are adjusted to this
                number if required.
            feature_map: The (parametrized) circuit to be used as a feature map for the underlying
                QNN. If ``None`` is given, the :class:`~qiskit.circuit.library.ZZFeatureMap`
                is used if the number of qubits is larger than 1. For a single qubit
                classification problem the :class:`~qiskit.circuit.library.ZFeatureMap`
                is used by default.
            ansatz: The (parametrized) circuit to be used as an ansatz for the underlying QNN.
                If ``None`` is given then the :class:`~qiskit.circuit.library.RealAmplitudes`
                circuit is used.
            loss: A target loss function to be used in training. Default value is ``cross_entropy``.
            optimizer: An instance of an optimizer or a callable to be used in training.
                Refer to :class:`~qiskit_algorithms.optimizers.Minimizer` for more information on
                the callable protocol. When `None` defaults to
                :class:`~qiskit_algorithms.optimizers.SLSQP`.
            warm_start: Use weights from previous fit to start next fit.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
            sampler: an optional Sampler primitive instance to be used by the underlying
                :class:`~qiskit_machine_learning.neural_networks.SamplerQNN` neural network. If
                ``None`` is passed then an instance of the reference Sampler will be used.
        Raises:
            QiskitMachineLearningError: Needs at least one out of ``num_qubits``, ``feature_map`` or
                ``ansatz`` to be given. Or the number of qubits in the feature map and/or ansatz
                can't be adjusted to ``num_qubits``.
        """

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

        neural_network = SamplerQNN(
            sampler=sampler,
            circuit=self._circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=self._get_interpret(2),
            output_shape=2,
            input_gradients=False,
        )

        super().__init__(
            neural_network=neural_network,
            loss=loss,
            one_hot=True,
            optimizer=optimizer,
            warm_start=warm_start,
            initial_point=initial_point,
            callback=callback,
        )

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the used feature map."""
        return self._feature_map

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the used ansatz."""
        return self._ansatz

    @property
    def circuit(self) -> QuantumCircuit:
        """Returns the underlying quantum circuit."""
        return self._circuit

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        return self.circuit.num_qubits

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> OptimizerResult:
        """
        Fit the model to data matrix X and targets y.

        Args:
            X: The input feature values.
            y: The input target values. Required to be one-hot encoded.

        Returns:
            Trained classifier.
        """
        X, y = self._validate_input(X, y)
        num_classes = self._num_classes

        # instance check required by mypy (alternative to cast)
        if isinstance(self._neural_network, SamplerQNN):
            self._neural_network.set_interpret(self._get_interpret(num_classes), num_classes)

        function = self._create_objective(X, y)
        return self._minimize(function)

    def _get_interpret(self, num_classes: int):
        def parity(x: int, num_classes: int = num_classes) -> int:
            return x % num_classes

        return parity

def _backward(
        self,
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray | SparseArray | None, np.ndarray | SparseArray | None]:
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        # freeze the first n parameters
        n = self._num_inputs  # assuming you want to freeze the first num_inputs parameters
        frozen_params = parameter_values[:, :n]
        trainable_params = parameter_values[:, n:]

        input_grad, weights_grad = None, None

        if np.prod(trainable_params.shape) > 0:
            circuits = [self._circuit] * num_samples

            job = None
            if self._input_gradients:
                job = self.gradient.run(circuits, trainable_params, parameters=self._circuit.parameters[n:])
            elif len(trainable_params[0]) > 0:
                params = [self._circuit.parameters[n:]] * num_samples
                job = self.gradient.run(circuits, trainable_params, parameters=params)

            if job is not None:
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError("Sampler job failed.") from exc

                input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        # set the gradients of the frozen parameters to zero
        if input_grad is not None:
            input_grad[:, :n] = 0
        if weights_grad is not None:
            weights_grad[:, :n] = 0

        return input_grad, weights_grad

