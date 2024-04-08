import numpy as np

from qiskit_machine_learning.utils.loss_functions import Loss, CrossEntropyLoss

from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, Minimizer

from qiskit_machine_learning.utils import  derive_num_qubits_feature_map_ansatz
from typing import Callable

from utility.froze import FrozenQNN   


class CrossDistilledLoss(Loss):
    def __init__(self, temperature=1.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = CrossEntropyLoss()

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