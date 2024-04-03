
import numpy as np
import scipy
from scipy import sparse
from qiskit_machine_learning.neural_networks import SamplerQNN

class Frozen(SamplerQNN):
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