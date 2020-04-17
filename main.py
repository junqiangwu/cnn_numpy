import numpy as np
from data import load_mnist
from pooling import AvgPooling,MaxPooling
from fc import FC
from conv import Conv2D
from module import Layers,Module
from activate import Relu,Sigmoid,Tanh
from loss import Softmax,CrossEntropyLoss,MSELoss

class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.layers = [
            # Conv2D(name="conv1",in_channels= 1, out_channels= 6,kernel_size=3,stride=1,padding=1), # 28
            # MaxPooling('pool1',ksize=2,stride=2), # 14
            # Relu(name='relu'),

            # Conv2D(name="conv2",in_channels= 6, out_channels= 12,kernel_size=3,stride=1,padding=1),
            # MaxPooling('pool2',ksize=2,stride=2), # 7*7*32
            # Relu(name='relu2'),

            FC(name="full1",in_channels= 28*28 , out_channels= 512),
            Tanh(name="sigmoid1"),
            FC(name="full2",in_channels=512,out_channels=128),
            Tanh(name="sigmoid2"),
            FC(name="full3",in_channels=128,out_channels=10),
        ]

def val(net,data,labels):
    n_correct = 0
    
    for i in range(0,data.shape[0],100):
        input = data[i:i+100].reshape(100,1,28,28)/255
        input = (input-0.5)/0.5
        yt = labels[i:i+100]

        output = net.forward(input)
        pred = Softmax(output)

        n_correct += np.sum(np.argmax(pred,axis=1) == yt)

    print("val test iter: {} acc: {:.3f} ".format(data.shape[0],n_correct/data.shape[0]))

def train():
    
    train_images, train_labels = load_mnist('./data/mnist')
    test_images, test_labels = load_mnist('./data/mnist', 't10k')

    net  = Net()
    loss = CrossEntropyLoss()

    for epoch in range(20):
        x = []
        y=[]
        n_correct = 0
        acc = 0
        for i in range(0,60000,100):
            input = train_images[i:i+100].reshape(100,1,28,28)/255
            input = (input-0.5)/0.5
            label = train_labels[i:i+100]

            output = net.forward(input)
            pred = Softmax(output)

            n_correct += np.sum(np.argmax(pred,axis=1) == label)

            loss_value,grad = loss(pred,label)

            net.backward(grad)
            net.step(lr=1e-3)
            
            acc = n_correct/(i+100)

            if i%(50*100) == 0:
                print("epoch: {} iter: {} loss: {} acc: {:.3f} n_correct: {}".format(epoch,i,loss_value,acc,n_correct))
    
        val(net,test_images,test_labels)


def main():
    train()





if __name__ == '__main__':
    main()