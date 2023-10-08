import torch
from torch import nn, optim
from tqdm import tqdm

from ResNet_model.resnet import get_resnet
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch.utils.data import Dataset

batch_size = 128
Train_Data = []
Test_Data = []
class_name = ['boxing','handclapping','handwaving','running','walking','jogging']
for index, each in enumerate(class_name):
    data_buff = []
    class_train = []
    class_test = []

    data_x = torch.load('./dataset/Pose_Dataset_Tensor/' + class_name[index] + '_data_x.pt')
    data_y = torch.load('./dataset/Pose_Dataset_Tensor/' + class_name[index] + '_data_y.pt')

    for i in range(len(data_x)):
        data_node = (data_x[i], data_y[i], index)
        data_buff.append(data_node)

    class_train = data_buff[:-50]
    class_test = data_buff[-50:]

    for each in class_train:
        Train_Data.append(each)
    for each in class_test:
        Test_Data.append(each)


# Dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.stack([self.data[index][0], self.data[index][1]], dim=0)
        label = self.data[index][2]
        return input_tensor, label

train_dataset = MyDataset(Train_Data)
test_dataset = MyDataset(Test_Data)

train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_resnet(pretrained=True).to(device)
print("# parameters:", sum(param.numel() for param in model.parameters()))

optimizer = optim.Adam(params=model.parameters(),lr=1e-2)

criterion = nn.CrossEntropyLoss()

total_epoch = 100

def train(epochs):
    train_accuracy, train_loss = 0.0, 0.0
    model.train()
    print('\nTrain start')
    for images, labels in tqdm(train_loader):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        out = model(images)

        loss = criterion(out, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = out.argmax(axis=1)

        train_accuracy += torch.sum(preds == labels).item() / len(labels)

    print(f"epoch: {epochs + 1}")
    print(f"train_loss: {train_loss / len(train_loader)}")
    print(f"train_accuracy: {train_accuracy / len(train_loader)}")
    with open('./ResNet_result/data25_train.csv', 'a') as f:
            f.write('{:<3d},{:<3f},{:<3f}\n'.format(epochs+1,train_loss / len(train_loader),train_accuracy / len(train_loader)))
    train_loss = train_loss / len(train_loader)
    train_accuracy = train_accuracy / len(train_loader)
    return train_loss, train_accuracy

def val(epochs):
    val_accuracy, val_loss = 0.0, 0.0   
    preds_list = []
    true_list = [] 
    data_list = []
    model.eval()
    print('\nValidation start')
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            val_loss += loss.item()
            preds = out.argmax(axis=1)
            preds_list += preds.detach().cpu().numpy().tolist()
            true_list += labels.detach().cpu().numpy().tolist()
            data_list.append(images.cpu())
            val_accuracy += torch.sum(preds == labels).item() / len(labels)

    print(f"epoch: {epochs + 1}")
    print(f"Validation loss: {val_loss / len(test_loader)}")
    print(f"Validation accuracy: {val_accuracy / len(test_loader)}")
    with open('./ResNet_result/data25_val.csv', 'a') as f:
            f.write('{:<3d},{:<3f},{:<3f}\n'.format(epochs+1,val_loss / len(test_loader),val_accuracy / len(test_loader)))
    val_loss = val_loss / len(test_loader)
    val_accuracy = val_accuracy / len(test_loader)
    return true_list, preds_list, data_list, val_loss, val_accuracy

def run():

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    
    for epochs in range(total_epoch):
        train_loss, train_accuracy =train(epochs)
        true_list, preds_list, data_list, val_loss, val_accuracy = val(epochs)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

    cm = confusion_matrix(true_list, preds_list)
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predict class', fontsize=13)
    plt.ylabel('True class', fontsize=13)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('./ResNet_result/confusion_matrix_25.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train')
    ax.plot(range(len(val_loss_list)), val_loss_list, c='r', label='validation')
    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('loss', fontsize='20')
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6])
    ax.set_title('train and validation loss', fontsize='20')
    ax.grid()
    ax.legend(fontsize='20')
    plt.show()
    plt.savefig('./ResNet_result/loss_graph_25.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(train_accuracy_list)), train_accuracy_list, c='b', label='train')
    ax.plot(range(len(val_accuracy_list)), val_accuracy_list, c='r', label='validation')
    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('accuracy', fontsize='20')
    ax.set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_title('train and validation accuracy', fontsize='20')
    ax.grid()
    ax.legend(fontsize='20')
    plt.show()
    plt.savefig('./ResNet_result/accuracy_graph_25.png')

    fig = plt.figure(figsize=(20,5))
    data_block = torch.cat(data_list,dim=0)
    idx_list = [n for n,(x,y) in enumerate(zip(true_list,preds_list)) if x!=y]
    len(idx_list)
    for i,idx in enumerate(idx_list[:20]):
        ax = fig.add_subplot(2,10,1+i)
        ax.axis('off')
        ax.set_title(f'true:{true_list[idx]} pred:{preds_list[idx]}')
        ax.imshow(data_block[idx,0])
        plt.savefig('./ResNet_result/misspreddata_25')

if __name__ == "__main__":
    run()