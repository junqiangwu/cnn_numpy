import numpy as np
from module import Layers

class MaxPooling(Layers):
    def __init__(self, name, ksize=2, stride=2):
        super(MaxPooling,self).__init__(name)
        self.ksize = ksize
        self.stride = stride 

    def forward(self, x):
        n,c,h,w = x.shape
        out = np.zeros([n, c, h//self.stride,w//self.stride])
        self.index = np.zeros_like(x)
        for b in range(n):
            for d in range(c):
                for i in range(h//self.stride):
                    for j in range(w//self.stride):
                        _x = i*self.stride
                        _y = j*self.stride
                        out[b, d ,i , j] = np.max(
                            x[b, d ,_x:_x+self.ksize, _y:_y+self.ksize])
                        index = np.argmax(x[b, d ,_x:_x+self.ksize, _y:_y+self.ksize])
                        self.index[b,d,_x+index//self.ksize, _y+index%self.ksize] = 1
        return out

    def backward(self, grad_out):
        return np.repeat(np.repeat(grad_out, self.stride, axis=2), self.stride, axis=3) * self.index

class AvgPooling(Layers):
    def __init__(self, name, ksize=2, stride=2):
        super(AvgPooling,self).__init__(name)
        self.ksize = ksize
        self.stride = stride 

    def forward(self, x):
        n,c,h,w = x.shape
        out = np.zeros([n, c, h//self.stride,w//self.stride])
        
        for b in range(n):
            for d in range(c):
                for i in range(h//self.stride):
                    for j in range(w//self.stride):
                        _x = i*self.stride
                        _y = j*self.stride
                        out[b, d ,i , j] = np.mean(
                            x[b, d ,_x:_x+self.ksize, _y:_y+self.ksize])
                        
        return out

    def backward(self, eta):
        next_eta = np.repeat(np.repeat(eta, self.stride, axis=2), self.stride, axis=3)
        return next_eta/(self.ksize*self.ksize)
