import numpy as np
import os
import logging

class Layers():
    def __init__(self,name):
        self.name = name
        
    def forward(self,x):
        pass
    def zero_grad(self):
        pass
    def backward(self,grad_out):
        pass
    def update(self,lr=1e-3):
        pass


class Module():
    def __init__(self):
        self.layers = []

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,grad):
        for layer in reversed(self.layers):
            layer.zero_grad()
            grad = layer.backward(grad)
    
    def step(self,lr=1e-3):
        for layer in reversed(self.layers):
            layer.update(lr)
