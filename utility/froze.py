
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

from qiskit_machine_learning.neural_networks.neural_network import NeuralNetwork
from qiskit_machine_learning.neural_networks.sampler_qnn import SamplerQNN
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

class FrozenQNN(SamplerQNN):
    

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
        print("hrllo")
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