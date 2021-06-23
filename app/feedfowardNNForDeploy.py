'''
MNIST
DataLoader, Transformation
Multilayer NN
Loss and optimizer
Training Loop
Model evaluation

'''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/mnist")

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size =784 # 28X28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size =100
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3082,))])
''' import mnist data'''
train_dateset = torchvision.datasets.MNIST(root='./data',train= True,
                                           transform=transform,download=True)

test_dateset = torchvision.datasets.MNIST(root='./data',train= False,
                                           transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dateset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dateset,batch_size=batch_size,
                                          shuffle=False)

examples  = iter(train_loader)
samples,lables = examples.next()
print(samples.shape, lables.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
    # plt.show()

img_grid=torchvision.utils.make_grid(samples)
# writer.add_image('mnist_images',img_grid)
# writer.close()


class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size , hidden_size,num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# writer.add_graph(model,samples.reshape(-1,28*28))
# writer.close()
# sys.exit()
''' training loop'''
n_total_steps = len(train_loader)
running_loss =0.0
running_correct =0
for epoch in range(num_epochs):
    for i , (images,lables) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        lables = lables.to(device)

        # foward pass
        outputs = model(images)
        loss = criterion(outputs,lables)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == lables).sum().item()

        if (i+1)%100 ==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps} , loss= {loss.item():.4f}')
            # writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps+i)
            # writer.add_scalar('accuracy', running_correct, epoch*n_total_steps+i)
            running_correct=0.0
            running_loss=0.0

# testing
preds =[]
labelslst =[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images= images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # values, index
        _,predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        class_predictions = [F.softmax(output,dim=0) for output in outputs]
        preds.append(class_predictions)
        labelslst.append(predictions)
    preds = torch.cat([torch.stack(batch) for batch in preds ])
    labelslst = torch.cat(labelslst)

    acc =100.0* (n_correct/n_samples)
    print(f' accuracy {acc}')
    classes = range(10)
    # for i in classes:
    #     lables_i = labelslst== i
    #     preds_i= preds[:,i]
        # writer.add_pr_curve(str(i),lables_i,preds_i,global_step=0)
        # writer.close()


torch.save(model.state_dict(),"mnist_ffn.pth")

