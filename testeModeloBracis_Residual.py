
import torch.utils.data as data_utils
import torch.optim as optm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import lr_scheduler
from fastai.text import *
import TreinaS11_semOverfit as tS11
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F


torch.nn.Module.dump_patches = True
print(torch.cuda.is_available())
device = 'cuda'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    print('sim')
    torch_seed = torch.initial_seed()
    # Numpy expects unsigned integer seeds.
    np_seed = torch_seed // 2 ** 32 - 1
    np.random.seed(np_seed)
except Exception:
    pass


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(32,32, (3,19))
        self.bn1 = nn.BatchNorm2d(32, momentum=0.99, affine=False)  #
        self.relu1 = nn.ReLU(inplace=True)
##
        self.conv2 = nn.Conv2d(32,32, (3,19))
        self.bn2 = nn.BatchNorm2d(32, momentum=0.99, affine=False)  #
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32,32, (3,19))

        self.avg1 = nn.AvgPool2d(2)#primeiro add
##
        self.bn3 = nn.BatchNorm2d(32, momentum=0.99, affine=False)  #
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, (3,19))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.99, affine=False)  #
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(32, 32, (3,19))

        self.avg2 = nn.AvgPool2d(2)#segundo add
##
        self.bn5 = nn.BatchNorm2d(32, momentum=0.99, affine=False)  #
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 64, (3,19))
        self.bn6 = nn.BatchNorm2d(64, momentum=0.99, affine=False)  #
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(0.2)
        self.conv7 = nn.Conv2d(64, 64, (3,19))

        self.avg3 = nn.AvgPool2d(2)  # terceiro add
##
        self.bn7 = nn.BatchNorm2d(64, momentum=0.99, affine=False)  #
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(64, 64, (3, 19))
        self.bn8 = nn.BatchNorm2d(64, momentum=0.99, affine=False)  #
        self.relu8 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.2)
        self.conv9 = nn.Conv2d(64, 64, (3, 19))

        self.avg4 = nn.AvgPool2d(2)  # quarto add
        ##
        self.bn9 = nn.BatchNorm2d(64, momentum=0.99, affine=False)  #
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(64, 128, (3, 19))
        self.bn10 = nn.BatchNorm2d(128, momentum=0.99, affine=False)  #
        self.relu10 = nn.ReLU(inplace=True)
        self.dropout210 = nn.Dropout(0.2)
        self.conv11 = nn.Conv2d(128, 128, (3, 19))

        self.avg5 = nn.AvgPool2d(2)  # quinto add
##
        self.bn11 = nn.BatchNorm2d(128, momentum=0.99, affine=False)  #
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(128, 128, (3, 19))
        self.bn12 = nn.BatchNorm2d(128, momentum=0.99, affine=False)  #
        self.relu12 = nn.ReLU(inplace=True)
        self.dropout12 = nn.Dropout(0.2)
        self.conv13 = nn.Conv2d(128, 128, (3, 19))

        self.avg6 = nn.AvgPool2d(2)  # segundo add
        ##
        self.bn13 = nn.BatchNorm2d(128, momentum=0.99, affine=False)  #
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(128, 256, (3, 19))
        self.bn14 = nn.BatchNorm2d(256, momentum=0.99, affine=False)  #
        self.relu14 = nn.ReLU(inplace=True)
        self.dropout14 = nn.Dropout(0.2)
        self.conv15 = nn.Conv2d(256, 256, (3, 19))

        self.avg7 = nn.AvgPool2d(2)  # terceiro add
        ##
        self.bn15 = nn.BatchNorm2d(256, momentum=0.99, affine=False)  #
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(256, 256, (3, 19))
        self.bn16 = nn.BatchNorm2d(256, momentum=0.99, affine=False)  #
        self.relu16 = nn.ReLU(inplace=True)
        self.dropout16 = nn.Dropout(0.2)
        self.conv17 = nn.Conv2d(256, 256, (3, 19))

        self.avg8 = nn.AvgPool2d(2)  # quarto add
        ##
        self.bn17 = nn.BatchNorm2d(256, momentum=0.99, affine=False)  #
        self.relu17 = nn.ReLU(inplace=True)
        self.conv18 = nn.Conv2d(256, 512, (3, 19))
        self.bn18 = nn.BatchNorm2d(512, momentum=0.99, affine=False)  #
        self.relu18 = nn.ReLU(inplace=True)
        self.dropout18 = nn.Dropout(0.2)
        self.conv19 = nn.Conv2d(512, 512, (3, 19))

        self.avg9 = nn.AvgPool2d(2)  # quinto add
        ##
        self.bn19 = nn.BatchNorm2d(512, momentum=0.99, affine=False)  #
        self.relu19 = nn.ReLU(inplace=True)
        self.conv20 = nn.Conv2d(512, 512, (3, 19))
        self.bn20 = nn.BatchNorm2d(512, momentum=0.99, affine=False)  #
        self.relu20 = nn.ReLU(inplace=True)
        self.dropout20 = nn.Dropout(0.2)
        self.conv21 = nn.Conv2d(512, 512, (3, 19))

        self.avg10 = nn.AvgPool2d(2)  # segundo add
        ##
        self.bn21 = nn.BatchNorm2d(512, momentum=0.99, affine=False)  #
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22= nn.Conv2d(512, 1024, (3, 19))
        self.bn22 = nn.BatchNorm2d(1024, momentum=0.99, affine=False)  #
        self.relu22 = nn.ReLU(inplace=True)
        self.dropout22 = nn.Dropout(0.2)
        self.conv23 = nn.Conv2d(1024, 1024, (3, 19))

        self.avg11 = nn.AvgPool2d(2)  # terceiro add
        ##
        self.bn23 = nn.BatchNorm2d(1024, momentum=0.99, affine=False)  #
        self.relu23 = nn.ReLU(inplace=True)
        self.conv24 = nn.Conv2d(1024, 1024, (3, 19))
        self.bn24 = nn.BatchNorm2d(1024, momentum=0.99, affine=False)  #
        self.relu24 = nn.ReLU(inplace=True)
        self.dropout24 = nn.Dropout(0.2)
        self.conv25 = nn.Conv2d(1024, 1024, (3, 19))

        self.avg12 = nn.AvgPool2d(2)  # quarto add
        ##
        self.bn25 = nn.BatchNorm2d(1024, momentum=0.99, affine=False)  #
        self.relu25 = nn.ReLU(inplace=True)
        self.conv26 = nn.Conv2d(1024, 2048, (3, 19))
        self.bn26 = nn.BatchNorm2d(2048, momentum=0.99, affine=False)  #
        self.relu26 = nn.ReLU(inplace=True)
        self.dropout26 = nn.Dropout(0.2)
        self.conv27 = nn.Conv2d(2048, 2048, (3, 19))

        self.avg13 = nn.AvgPool2d(2)  # quinto add
        ##
        self.bn27 = nn.BatchNorm2d(2048, momentum=0.99, affine=False)  #
        self.relu27 = nn.ReLU(inplace=True)
        self.conv28 = nn.Conv2d(2048, 2048, (3, 19))
        self.bn28 = nn.BatchNorm2d(2048, momentum=0.99, affine=False)  #
        self.relu28 = nn.ReLU(inplace=True)
        self.dropout28 = nn.Dropout(0.2)
        self.conv29 = nn.Conv2d(2048, 2048, (3, 19))

        self.avg14 = nn.AvgPool2d(2)  # segundo add
        ##
        self.bn29 = nn.BatchNorm2d(2048, momentum=0.99, affine=False)  #
        self.relu29 = nn.ReLU(inplace=True)
        self.conv30 = nn.Conv2d(2048, 4096, (3, 19))
        self.bn30 = nn.BatchNorm2d(4096, momentum=0.99, affine=False)  #
        self.relu30 = nn.ReLU(inplace=True)
        self.dropout30 = nn.Dropout(0.2)
        self.conv31 = nn.Conv2d(4096, 4096, (3, 19))

        self.avg15 = nn.AvgPool2d(2)  # terceiro add
        ##
        self.bn31 = nn.BatchNorm2d(4096, momentum=0.99, affine=False)  #
        self.relu31 = nn.ReLU(inplace=True)
        self.conv32 = nn.Conv2d(4096, 4096, (3, 19))
        self.bn32 = nn.BatchNorm2d(4096, momentum=0.99, affine=False)  #
        self.relu32 = nn.ReLU(inplace=True)
        self.dropout32 = nn.Dropout(0.2)
        self.conv33 = nn.Conv2d(4096, 4096, (3, 19))

        self.avg16 = nn.AvgPool2d(2)  # segundo add

        self.bn33 = nn.BatchNorm2d(4096, momentum=0.99, affine=False)  #
        self.relu33 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(4096, 2)


    def forward(self, x, pj):#  pj vetor com 0 onde nao deve ter dropout

        x = self.relu1(self.bn1(self.conv1(x)))
        input1 = x

        x = self.relu2(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv3(x)

        ay = self.avg1(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input1 + ay
        input2 = xSum

        x = self.relu3(self.bn3(xSum))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x =self.conv5(x)

        ay = self.avg2(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input2 + ay
        input3 = xSum

        x = self.relu5(self.bn5(xSum))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x =self.conv7(x)

        ay = self.avg3(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input3 + ay
        input4 = xSum

        x = self.relu7(self.bn7(xSum))
        x = self.relu8(self.bn8(self.conv8(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv9(x)

        ay = self.avg4(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input4 + ay
        input5 = xSum

        x = self.relu9(self.bn9(xSum))
        x = self.relu10(self.bn10(self.conv10(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv11(x)

        ay = self.avg5(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input5 + ay
        input6 = xSum

        x = self.relu11(self.bn11(xSum))
        x = self.relu12(self.bn12(self.conv10(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv13(x)

        ay = self.avg6(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input5 + ay
        input6 = xSum

        x = self.relu13(self.bn13(xSum))
        x = self.relu14(self.bn14(self.conv14(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv15(x)

        ay = self.avg7(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input6 + ay
        input7 = xSum

        x = self.relu15(self.bn15(xSum))
        x = self.relu16(self.bn16(self.conv16(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv17(x)

        ay = self.avg8(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input7 + ay
        input8 = xSum

        x = self.relu17(self.bn17(xSum))
        x = self.relu18(self.bn18(self.conv18(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv19(x)

        ay = self.avg9(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input8 + ay
        input9 = xSum

        x = self.relu19(self.bn19(xSum))
        x = self.relu20(self.bn20(self.conv20(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv21(x)

        ay = self.avg10(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input9 + ay
        input10 = xSum

        x = self.relu21(self.bn21(xSum))
        x = self.relu22(self.bn22(self.conv22(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv23(x)

        ay = self.avg11(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input10 + ay
        input11 = xSum

        x = self.relu23(self.bn23(xSum))
        x = self.relu24(self.bn24(self.conv24(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv25(x)

        ay = self.avg12(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input11 + ay
        input12 = xSum

        x = self.relu25(self.bn25(xSum))
        x = self.relu26(self.bn26(self.conv26(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv27(x)

        ay = self.avg13(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input12 + ay
        input13 = xSum

        x = self.relu28(self.bn28(xSum))
        x = self.relu29(self.bn29(self.conv29(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv30(x)

        ay = self.avg14(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input13 + ay
        input14 = xSum

        x = self.relu30(self.bn30(xSum))
        x = self.relu31(self.bn31(self.conv31(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv32(x)

        ay = self.avg15(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input14 + ay
        input15 = xSum

        x = self.relu32(self.bn32(xSum))
        x = self.relu33(self.bn33(self.conv31(x)))
        x = F.dropout(x, p=0.2, training=self.training)  # self.dropout(x)#
        x = self.conv33(x)

        ay = self.avg16(x.unsqueeze(1))  # (7,7) #ay torch.Size([256, 1, 57])
        ay = ay.view([ay.shape[0], ay.shape[2]])  # torch.Size([256, 57])
        xSum = input15 + ay

        x =  self.fc8(xSum)

        return F.softmax(x, dim=1)  # x




def create_datasets(batch_size, train, train_size, valid_size):
    ####################
    #  obtain training indices that will be used for validation

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    trai = round(num_train*train_size)
    val = round(num_train*valid_size)

    train_idx, valid_idx, test_idx  = indices[:trai], indices[trai:trai+val], indices[trai+val:]


    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(valid_idx)

    train_loader = data_utils.DataLoader(dataset=train, batch_size=batch_size, sampler=train_sampler)
    valid_loader = data_utils.DataLoader(dataset=train, batch_size=batch_size, sampler=valid_sampler)
    test_loader = data_utils.DataLoader(dataset=train, batch_size=batch_size, sampler=test_sampler)


    return train_loader, valid_loader, test_loader




# mat = scipy.io.loadmat(
#     'D:/CNN_BomRuim/baseExcRuim_comClasseBin.mat',squeeze_me=True, struct_as_record=False)
import h5py
import numpy as np
# filepath = 'D:/CNN_BomRuim/base4D.mat'
# X_train = {}
# f = h5py.File(filepath)
# for k, v in f.items():
#     X_train[k] = np.array(v)

# filepath = 'D:/CNN_BomRuim/classeBin.mat'
# Y_train = {}
# f = h5py.File(filepath)
# for k, v in f.items():
#     Y_train[k] = np.array(v)

f = h5py.File('D:/CNN_BomRuim/base4D.mat','r')
data = f.get('base')
data = np.array(data) # For converting to a NumPy array

f = h5py.File('D:/CNN_BomRuim/classeBin.mat','r')
classe = f.get('classeBin')
classe = np.array(classe) # For converting to a NumPy array


num_classes = 2
valid_size = 0.2
batch_size = 16
train_size = 0.7


X_train = torch.from_numpy((data))
Y_train = torch.from_numpy(np.transpose(classe))

print(X_train.size(0))
print(Y_train.size(0))

train = data_utils.TensorDataset(X_train.float(), Y_train.long())
train_loader, valid_loader, test_loader = create_datasets(batch_size, train, train_size, valid_size)


model = NeuralNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optm.Adam(model.parameters(), lr=1e-2)
exp_lr_scheduler = ReduceLROnPlateau(optimizer)
lr_schedul = lr_scheduler.StepLR(optimizer,step_size=10)
exp_lr_scheduler = ReduceLROnPlateau(optimizer, verbose=True,
                                     cooldown=50)

tS11.train_Valid(model, 50, 100, exp_lr_scheduler, train_loader,
                      valid_loader, optimizer, criterion, lr_schedul,
                                   'checkpoint27layers_weight_decay=_t1_exp1.pt')

result = []
with torch.no_grad():
    pred = []
    correct = 0
    total = 0
    # images, labels in data:
    for images, labels in test_loader:  # enumerate(data, 1):
        images = images.float()
        labels = labels.long()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images, pj)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred.append(predicted)

    result = torch.cat(pred, dim=-1)
    acc = 100 * correct / total
    print('Accuracy of the network on the test data: {} %'.format(acc))

torch.save(model.state_dict(),'teste_t1_exp1-epoch1_2.pth')
np.save('teste_t1_exp1-epoch1_2.npy',Tensor.cpu(result))
