import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import torchnet as tnt
from PIL import Image
import xlwt

def ArrayTopPicture(array):
    image = Image.fromarray(array)
    return image

work_book = xlwt.Workbook(encoding='utf-8')

work_sheet = work_book.add_sheet('sheet1', cell_overwrite_ok=True)

PATH = './epochs_test/epoch_100.pt'
NUM_ROUTING_ITERATIONS = 4
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS,flag = None):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules
        self.flag = flag
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (0.5 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                if i == NUM_ROUTING_ITERATIONS-1 :
                    aa = probs.squeeze()

                    for i in range(6):
                        for j in range(64):
                            data_w = aa[i][0][j][0].item()

                            if self.flag == 1:
                                work_sheet.write(i, j, label=str(data_w))

                            if self.flag == 2:
                                work_sheet.write(i+7, j, label=str(data_w))

                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs

class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        self.primary_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=-1, in_channels=128, out_channels=8,
                                             kernel_size=9, stride=2)

        self.digit_capsules_1 = CapsuleLayer(num_capsules=6, num_route_nodes=4 * 4 * 4, in_channels=16,
                                           out_channels=32, flag = 1)
        self.digit_capsules_2 = CapsuleLayer(num_capsules=6, num_route_nodes=4 * 4 * 4, in_channels=16,
                                           out_channels=32, flag = 2)

        self.decoder = nn.Sequential(
            nn.Linear(2 * 32 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 289 * 2),

        )

        # self.decoder_1 = nn.Sequential(
        #     nn.Linear(32 * NUM_CLASSES, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 289),
        #     # nn.Sigmoid()
        #     # nn.ReLU(inplace=True)
        # )
        #
        # self.decoder_2 = nn.Sequential(
        #     nn.Linear(32 * NUM_CLASSES, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 289),
        #     # nn.Sigmoid()
        #     # nn.ReLU(inplace=True)
        # )

    def forward(self, x, y=None):

        x = self.conv(x)

        x = self.primary_capsules(x)

        x_1, x_2 = x.split(64, dim=1)

        x_1 = self.digit_capsules_1(x_1).squeeze().transpose(0, 1)
        x_2 = self.digit_capsules_2(x_2).squeeze().transpose(0, 1)

        # x = self.digit_capsules_1(x_1).squeeze().transpose(0, 1)

        # x = torch.stack([x_1,x_2],dim=2)
        # class_x = torch.matmul(x.transpose(2, 3),x)

        classes = ((x_1 ** 2).sum(dim=-1) + (x_2 ** 2).sum(dim=-1))** 0.5
        classes = F.softmax(classes, dim=-1)


        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(6)).cuda().index_select(dim=0, index=max_length_indices.data)

        r1 = (x_1 * y[:, :, None]).contiguous().view(x_1.size(0), -1)
        r2 = (x_2 * y[:, :, None]).contiguous().view(x_2.size(0), -1)
        r = torch.stack([r1, r2], dim=1).contiguous().view(r1.size(0), -1)

        # reconstructions_1 = self.decoder_1((x_1 * y[:, :, None]).contiguous().view(x_1.size(0), -1))
        # reconstructions_2 = self.decoder_2((x_2 * y[:, :, None]).contiguous().view(x_2.size(0), -1))

        # reconstructions = torch.stack([reconstructions_1, reconstructions_2], dim=1)

        reconstructions = self.decoder(r)

        return classes, reconstructions

Train_Data = []
Test_Data = []
Train_Label = []
Test_Label = []

# class_name = ['boxing','handclapping','handwaving','running','walking','jogging']
class_name = ['jogging']
for index, each in enumerate(class_name):
    data_buff = []
    label_buff = []

    data_x = torch.load('./dataset/Pose_Dataset_Tensor/' + class_name[index] + '_data_x.pt')
    data_y = torch.load('./dataset/Pose_Dataset_Tensor/' + class_name[index] + '_data_y.pt')

    for i in range(len(data_x)):
        data_node = [data_x[i].tolist(), data_y[i].tolist()]

        data_buff.append(data_node)
        label_buff.append(index)

    class_data_train = data_buff[:-100]
    class_label_train = label_buff[:-100]
    class_data_test = data_buff[-100:]
    class_label_test = label_buff[-100:]

    for each in class_data_train:
        Train_Data.append(each)
    for each in class_label_train:
        Train_Label.append(each)
    for each in class_data_test:
        Test_Data.append(each)
    for each in class_label_test:
        Test_Label.append(each)

train_data  = torch.tensor(np.array(Train_Data))
test_data   = torch.tensor(np.array(Test_Data))
train_label = torch.tensor(np.array(Train_Label))
test_label  = torch.tensor(np.array(Test_Label))

data = test_data
labels = test_label
tensor_dataset = tnt.dataset.TensorDataset([data, labels])

a = tensor_dataset.parallel(batch_size=20, num_workers=0, shuffle=False)
test_sample = next(iter(a))

model = CapsuleNet()
model.cuda()
model.load_state_dict(torch.load(PATH))

ground_truth = (test_sample[0].float() / 255.0)

_, reconstructions = model(Variable(ground_truth).cuda())

work_book.save('coe_data.xls')