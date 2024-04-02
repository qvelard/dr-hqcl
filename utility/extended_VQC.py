
from abc import ABC, abstractmethod
import numpy as np

#from qiskit_machine_learning.neural_networks import SamplerQNN

#from qiskit_machine_learning.utils.loss_functions import Loss
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from qiskit import QuantumCircuit

from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, Minimizer

import torch
import torch.nn as nn

class Loss(ABC):
    """
    Abstract base class for computing Loss.
    """

    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        This method calls the ``evaluate`` method. This is a convenient method to compute loss.
        """
        return self.evaluate(predict, target)

    @abstractmethod
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        An abstract method for evaluating the loss function. Inputs are expected in a shape
        of ``(N, *)``. Where ``N`` is a number of samples. Loss is computed for each sample
        individually.

        Args:
            predict: an array of predicted values using the model.
            target: an array of the true values.

        Returns:
            An array with values of the loss function of the shape ``(N, 1)``.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match
        """
        raise NotImplementedError

    @staticmethod
    def _validate_shapes(predict: np.ndarray, target: np.ndarray) -> None:
        """
        Validates that shapes of both parameters are identical.

        Args:
            predict: an array of predicted values using the model
            target: an array of the true values

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """

        if predict.shape != target.shape:
            raise QiskitMachineLearningError(
                f"Shapes don't match, predict: {predict.shape}, target: {target.shape}!"
            )

    @abstractmethod
    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        An abstract method for computing the gradient. Inputs are expected in a shape
        of ``(N, *)``. Where ``N`` is a number of samples. Gradient is computed for each sample
        individually.

        Args:
            predict: an array of predicted values using the model.
            target: an array of the true values.

        Returns:
            An array with gradient values of the shape ``(N, *)``. The output shape depends on
            the loss function.

        Raises:
            QiskitMachineLearningError: shapes of predict and target do not match.
        """
        raise NotImplementedError


class CrossDistilledLoss(Loss):
    def __init__(self, temperature=1.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def evaluate(self, student_output: np.ndarray, teacher_output: np.ndarray, target: np.ndarray, is_new_data: np.ndarray) -> np.ndarray:
        self._validate_shapes(student_output, target)
        self._validate_shapes(teacher_output, target)
        self._validate_shapes(is_new_data, target)

        loss = np.zeros_like(target)

        # Classification loss for new data
        ce_loss = self.ce_loss.evaluate(student_output, target)
        loss += is_new_data * ce_loss

        # Distillation loss for old data
        teacher_output_soft = np.exp(teacher_output / self.temperature) / np.sum(np.exp(teacher_output / self.temperature), axis=-1, keepdims=True)
        student_output_soft = np.exp(student_output / self.temperature) / np.sum(np.exp(student_output / self.temperature), axis=-1, keepdims=True)

        kl_loss = np.sum(student_output_soft * (np.log(student_output_soft) - np.log(teacher_output_soft)), axis=-1, keepdims=True) * (self.temperature ** 2)
        loss += (1 - is_new_data) * kl_loss

        # Combine losses
        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        return loss

    def gradient(self, student_output: np.ndarray, teacher_output: np.ndarray, target: np.ndarray, is_new_data: np.ndarray) -> np.ndarray:
        self._validate_shapes(student_output, target)
        self._validate_shapes(teacher_output, target)
        self._validate_shapes(is_new_data, target)

        gradient = np.zeros_like(student_output)

        # Classification gradient for new data
        ce_gradient = self.ce_loss.gradient(student_output, target)
        gradient += is_new_data[:, np.newaxis] * ce_gradient

        # Distillation gradient for old data
        teacher_output_soft = np.exp(teacher_output / self.temperature) / np.sum(np.exp(teacher_output / self.temperature), axis=-1, keepdims=True)
        student_output_soft = np.exp(student_output / self.temperature) / np.sum(np.exp(student_output / self.temperature), axis=-1, keepdims=True)

        kl_gradient = (student_output_soft - teacher_output_soft) / self.temperature
        gradient += (1 - is_new_data)[:, np.newaxis] * kl_gradient

        # Combine gradients
        gradient = self.alpha * ce_gradient + (1 - self.alpha) * kl_gradient
          
        return gradient




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

