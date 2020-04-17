import numpy as np
from module import Layers

class FC(Layers):
    def __init__(self,name,in_channels,out_channels):
        super(FC,self).__init__(name)
        self.weights = np.random.standard_normal((in_channels,out_channels))
        self.bias = np.zeros(out_channels)

        self.grad_w = np.zeros((in_channels,out_channels))
        self.grad_b = np.zeros(out_channels)

    def forward(self, input):
        self.in_shape = input.shape 
        input = np.reshape(input,(input.shape[0],-1)) # flat
        self.input = input
        return np.dot(input,self.weights)+self.bias

    def backward(self,grad_out):
        N = grad_out.shape[0]
        dx = np.dot(grad_out,self.weights.T)

        self.grad_w = np.dot(self.input.T,grad_out)
        self.grad_b = np.sum(grad_out,axis=0)

        return dx.reshape(self.in_shape)
    
    def zero_grad(self):
        self.grad_w.fill(0)
        self.grad_b.fill(0)

    def update(self,lr=1e-3):
        self.weights -= lr*self.grad_w  
        self.bias -= lr*self.grad_b


if __name__ == '__main__':
    # 模拟 实际权重学习  y = wx+b
    w = np.random.randn(100,1)
    b = np.random.randn(1)

    x_data = np.random.randn(500,100)
    label = np.dot(x_data,w)+b

    layer = FC('fc',100,1)

    for i in range(10000):
        index = i%(500-10)
        x = x_data[index:index+10]
        y = label[index:index+10]

        out = layer.forward(x)

        loss = np.mean(np.sum(np.square(out-y),axis=-1))
        
        dy = out-y
        layer.zero_grad()
        grad = layer.backward(dy)
        layer.update(1e-3)

        if i%1000==0:
            print(loss)
