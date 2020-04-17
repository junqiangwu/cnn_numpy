
import numpy as np
from module import Layers

class Conv2D(Layers):
    def __init__(self,name,in_channels,out_channels, kernel_size, stride,padding,bias=True):
        super(Conv2D,self).__init__(name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.standard_normal((out_channels,in_channels,kernel_size,kernel_size))
        self.bias = np.zeros(out_channels)
        
        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.bias.shape)
    
    def _sing_conv(self,x):
        x = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant',constant_values=0)
        b,c,h,w = x.shape
        oh = (h-self.ksize)//self.stride +1
        ow = (w-self.ksize)//self.stride +1
        out = np.zeros((b,self.out_channels,oh,ow))
        for n in range(b):
            for d in range(self.out_channels):
                for i in range(0,oh,1):
                    for j in range(0,ow,1):
                        _x = i*self.stride
                        _y = j*self.stride
                        out[n,d,i,j] = np.sum(x[n,:,_x:_x+self.ksize,_y:_y+self.ksize]*self.weights[d,:,:,:])#+self.bias[d]
        return out
    
    def forward(self,x):
        self.x = x
        weights = self.weights.reshape(self.out_channels,-1) # o,ckk

        x = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant',constant_values=0)
        b,c,h,w = x.shape

        self.out = np.zeros((b,self.out_channels,(h-self.ksize)//self.stride +1,(w-self.ksize)//self.stride +1))
        
        self.col_img = self.im2col(x,self.ksize,self.stride) # bhw * ckk
        out = np.dot(weights,self.col_img.T).reshape(self.out_channels,b,-1).transpose(1,0,2)
        
        self.out = np.reshape(out,self.out.shape)

        return self.out

    def backward(self,grad_out):
        b,c,h,w = self.out.shape  # 
        
        grad_out_ = grad_out.transpose(1,0,2,3) #b,oc,h,w * (bhw , ckk)
        grad_out_flat = np.reshape(grad_out_, [self.out_channels, -1])
        
        self.grad_w = np.dot(grad_out_flat,self.col_img).reshape(self.grad_w.shape)
        self.grad_b = np.sum(grad_out_flat,axis=1)
        tmp = self.ksize - self.padding - 1
        grad_out_pad = np.pad(grad_out,((0,0),(0,0),(tmp,tmp),(tmp,tmp)),'constant',constant_values=0)
        
        flip_weights = np.flip(self.weights, (2, 3))
        # flip_weights = np.flipud(np.fliplr(self.weights)) # rot(180)
        flip_weights = flip_weights.swapaxes(0,1) # in oc
        col_flip_weights = flip_weights.reshape([self.in_channels,-1])
        
        weights = self.weights.transpose(1,0,2,3).reshape(self.in_channels,-1)
        
        col_grad = self.im2col(grad_out_pad,self.ksize,1) #bhw,ckk
        
        # (in,ckk) * (bhw,ckk).T 
        next_eta = np.dot(weights,col_grad.T).reshape(self.in_channels,b,-1).transpose(1,0,2) 
        
        next_eta = np.reshape(next_eta, self.x.shape)
       
        return next_eta

    def zero_grad(self):
        self.grad_w = np.zeros_like(self.grad_w)
        self.grad_b = np.zeros_like(self.grad_b)

    def update(self,lr=1e-3):
        self.weights -= lr*self.grad_w  
        self.bias -= lr*self.grad_b

    def im2col(self,x,k_size,stride):
        
        b,c,h,w = x.shape
        image_col = []
        for n in range(b):
            for i in range(0,h-k_size+1,stride):
                for j in range(0,w-k_size+1,stride):
                    col = x[n,:,i:i+k_size,j:j+k_size].reshape(-1)
                    image_col.append(col)
        
        return np.array(image_col)



if __name__ == '__main__':
    x = np.random.randn(5,3,32,32)

    conv = Conv2D('conv1',3,12,4,1,1)

    y = conv.forward(x)
    print(y.shape)

    loss = y-(y+1)
    grad = conv.backward(loss)

    print(grad.shape)

    # print(x,'\n\n',y)