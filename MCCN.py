import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 20
NUM_CLASSES = 6
NUM_EPOCHS = 200   #500
NUM_ROUTING_ITERATIONS = 4

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    #random shift positive（+n）：0 -> len-n replace n -> len
    #random shift negative（-n）：n -> len replace 0 -> len-n
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

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


        self.digit_capsules_1 = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=4 * 4 * 4, in_channels=16,
                                           out_channels=32)
        self.digit_capsules_2 = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=4 * 4 * 4, in_channels=16,
                                           out_channels=32)


        self.decoder = nn.Sequential(
            nn.Linear(2 * 32 * NUM_CLASSES, 512),
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
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        r1 = (x_1 * y[:, :, None]).contiguous().view(x_1.size(0), -1)
        r2 = (x_2 * y[:, :, None]).contiguous().view(x_2.size(0), -1)
        r = torch.stack([r1, r2], dim=1).contiguous().view(r1.size(0), -1)

        # reconstructions_1 = self.decoder_1((x_1 * y[:, :, None]).contiguous().view(x_1.size(0), -1))
        # reconstructions_2 = self.decoder_2((x_2 * y[:, :, None]).contiguous().view(x_2.size(0), -1))

        # reconstructions = torch.stack([reconstructions_1, reconstructions_2], dim=1)

        reconstructions = self.decoder(r)

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstructions = reconstructions.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


# Dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Concatenate two matrices in a tuple together as an input tensor
        input_tensor = torch.stack([self.data[index][0], self.data[index][1]], dim=0)
        # input_tensor = self.data[index][0]
        label = self.data[index][2]

        return input_tensor, label


Train_Data = []
Test_Data = []
Train_Label = []
Test_Label = []

class_name = ['boxing','handclapping','handwaving','running','walking','jogging']
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

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from tqdm import tqdm
    import torchnet as tnt


    model = CapsuleNet()

    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters(), lr=0.001)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()


    def get_iterator(mode):

        if mode:
            data = train_data
            labels = train_label
        else:
            data = test_data
            labels = test_label

        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    def processor(sample):
        data, labels, training = sample

        data = augmentation(data.float() / 255.0)
        labels = torch.Tensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.Tensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.Tensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs_test/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0].float() / 255.0)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data

        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
