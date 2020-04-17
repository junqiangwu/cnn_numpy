# cnn_numpy

使用numpy实现神经网络，在mnist上进行训练、测试

目前包括已下算子:

1. 卷积 

2. 全连接 FC

3. 池化  
- MaxPolling
- AvgPolling


4. 激活函数 
- Sigmoid
- Relu
- Tanh
- Softmax

5. 损失函数
- CE 交叉熵损失
- MSE 均方根损失



构建网络例子
```py
# 构建一个 conv+fc 的网络
class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.layers = [
            Conv2D(name="conv1",in_channels= 1, out_channels= 6,kernel_size=3,stride=1,padding=1), # 28
            MaxPooling('pool1',ksize=2,stride=2), # 14
            Tanh(name='relu'),

            Conv2D(name="conv2",in_channels= 6, out_channels= 12,kernel_size=3,stride=1,padding=1),
            MaxPooling('pool2',ksize=2,stride=2), # 7*7*32
            Tanh(name='relu2'),

            FC(name="full1",in_channels= 12*7*7 , out_channels= 512),
            Tanh(name="sigmoid1"),
            FC(name="full2",in_channels=512,out_channels=128),
            Tanh(name="sigmoid2"),
            FC(name="full3",in_channels=128,out_channels=10),
        ]

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

```

## 运行工程
1. 下载项目
`git clone git@github.com:junqiangwu/cnn_numpy.git`

2. 运行
`python3 main.py` # 默认使用mnist训练集

- 使用全连接层训练可以达到92%的准确率，这个使用了全部数据集 6w训练  1w测试

- 使用 Conv+FC 训练, 因为训练特别慢，只用了1w训练集进行训练, 训练5个epoch后,在测试集上达到**70**准确率，验证了代码的有效性

> 训练优化器使用基本的SGD，默认使用的1e-3初始学习率，使用过大学习率的话会学飞，有时间尝试一下Adam;


```bash
# log print
epoch: 19 iter: 57500 loss: 0.2737244704525886 acc: 0.902 n_correct: 51956
epoch: 19 iter: 58000 loss: 0.1859380691846724 acc: 0.902 n_correct: 52417
epoch: 19 iter: 58500 loss: 0.2722464711524591 acc: 0.903 n_correct: 52894
epoch: 19 iter: 59000 loss: 0.06477324861968438 acc: 0.903 n_correct: 53369
epoch: 19 iter: 59500 loss: 0.08215275528987825 acc: 0.903 n_correct: 53844

```

# TODO

- 添加优化器算法