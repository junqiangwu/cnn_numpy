from module import Layers
import numpy as np


class Relu(Layers):
    def __init__(self,name):
        super(Relu,self).__init__(name)

    def forward(self,input):
        self.input = input
        return np.maximum(input, 0)

    def backward(self,grad_out):
        grad_out[self.input<0]=0
        return grad_out

class Sigmoid(Layers):
    def __init__(self, name):
        super(Sigmoid,self).__init__(name)
    def forward(self,input):
        self.output = 1/(1+np.exp(-input))
        return self.output
    def backward(self,grad):
        grad = grad * self.output*(1-self.output)
        return grad

class Tanh(Layers):
    def __init__(self, name):
        super(Tanh,self).__init__(name)
    def forward(self,input):
        a = np.exp(input)
        b = np.exp(-input)
        self.output = (a-b)/(a+b)
        return self.output
    def backward(self,grad):
        grad = grad * (1-self.output*self.output)
        return grad