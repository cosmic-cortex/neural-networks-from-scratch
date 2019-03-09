import math
import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def gradX(self, x):
        pass

    @abstractmethod
    def gradW(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, Dy):
        pass
