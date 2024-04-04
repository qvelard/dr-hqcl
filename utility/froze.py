
"""A Neural Network implementation based on the Sampler primitive."""

from __future__ import annotations
import logging

from numbers import Integral
from typing import Callable, cast, Iterable, Sequence

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler, SamplerResult, Sampler
from qiskit_algorithms.gradients import (
    BaseSamplerGradient,
    ParamShiftSamplerGradient,
    SamplerGradientResult,
)

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
import qiskit_machine_learning.optionals as _optionals

from .neural_network import NeuralNetwork

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


logger = logging.getLogger(__name__)

class Frozen(SamplerQNN):
    def __init__(
        self,
        *,
        circuit: QuantumCircuit,
        sampler: BaseSampler | None = None,
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
        sparse: bool = False,
        interpret: Callable[[int], int | tuple[int, ...]] | None = None,
        output_shape: int | tuple[int, ...] | None = None,
        gradient: BaseSamplerGradient | None = None,
        input_gradients: bool = False,
    ):
        """
        Args:
            sampler: The sampler primitive used to compute the neural network's results.
                If ``None`` is given, a default instance of the reference sampler defined
                by :class:`~qiskit.primitives.Sampler` will be used.
            circuit: The parametrized quantum circuit that generates the samples of this network.
                If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is passed, the
                `input_params` and `weight_params` do not have to be provided, because these two
                properties are taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`.
            input_params: The parameters of the circuit corresponding to the input. If a
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `input_params` value here is ignored. Instead the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` input_parameters.
            weight_params: The parameters of the circuit corresponding to the trainable weights. If
                a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `weight_params` value here is ignored. Instead the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` weight_parameters.
            sparse: Returns whether the output is sparse or not.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If no interpret function is
                passed, then an identity function will be used by this neural network.
            output_shape: The output shape of the custom interpretation. It is ignored if no custom
                interpret method is provided where the shape is taken to be
                ``2^circuit.num_qubits``.
            gradient: An optional sampler gradient to be used for the backward pass.
                If ``None`` is given, a default instance of
                :class:`~qiskit_algorithms.gradients.ParamShiftSamplerGradient` will be used.
            input_gradients: Determines whether to compute gradients with respect to input data.
                 Note that this parameter is ``False`` by default, and must be explicitly set to
                 ``True`` for a proper gradient computation when using
                 :class:`~qiskit_machine_learning.connectors.TorchConnector`.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        # set primitive, provide default
        if sampler is None:
            sampler = Sampler()
        self.sampler = sampler

        # set gradient
        if gradient is None:
            gradient = ParamShiftSamplerGradient(self.sampler)
        self.gradient = gradient

        self._org_circuit = circuit

        if isinstance(circuit, QNNCircuit):
            self._input_params = list(circuit.input_parameters)
            self._weight_params = list(circuit.weight_parameters)
        else:
            self._input_params = list(input_params) if input_params is not None else []
            self._weight_params = list(weight_params) if weight_params is not None else []

        if sparse:
            _optionals.HAS_SPARSE.require_now("DOK")

        self.set_interpret(interpret, output_shape)
        self._input_gradients = input_gradients

        super().__init__(
            num_inputs=len(self._input_params),
            num_weights=len(self._weight_params),
            sparse=sparse,
            output_shape=self._output_shape,
            input_gradients=self._input_gradients,
        )

        if len(circuit.clbits) == 0:
            circuit = circuit.copy()
            circuit.measure_all()
        self._circuit = self._reparameterize_circuit(circuit, input_params, weight_params)
        
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