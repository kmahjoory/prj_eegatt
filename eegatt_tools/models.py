import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
cos = nn.CosineSimilarity(dim=2)
from torchsummary import summary


# In this model, EEG channels are split into 8 groups, first 4 corresponding to the one hemisphere & and the next 4 to the
# other hemisphere.
# In the first CNN layer, a 1dconv is applied for each group of channels separately! to extract temporal structure.
# In the 2nd CNN layer, a 1dconv is applied for each pair of channels resulting in 8*8/2
class CnnCos1layer(nn.Module):

    def __init__(self, n_inpch=[5, 6, 7, 3, 5, 6, 7, 3], n_inpd2=5*64, n_outs=2, ks1=9, ks2=3, n_ch1=1, n_fc1=8):
        super(CnnCos1layer, self).__init__()
        self.n_inpch = n_inpch
        self.conv1 = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(n_inpch[j], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
                           nn.ReLU()) for j in range(8)])
        self.fc = nn.Linear(int((8 * 7) / 2), n_outs)

    def forward(self, x):
        n_inpch = self.n_inpch
        indx_chans = [range(sum(n_inpch[:k]), sum(n_inpch[:k + 1])) for k in range(len(n_inpch))]
        out1, out2 = [], []
        # 42 Channels ->> 8 channel groups
        for j in range(8):
            out1.append(self.conv1[j](x[:, indx_chans[j], :]))
        out1 = torch.cat(out1, 1)
        similarity = [cos(out1[:, j, :], out1[:, k, :]).unsqueeze(dim=1) for j in range(8) for k in range(j+1, 8)]
        similarity = torch.cat(similarity, 1)
        out = similarity.view(similarity.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc(out)
        # out = torch.softmax(out, dim=1)  # classes are in columns
        return out#, similarity.mean(dim=0)


def weight_init(model):
    # set CNN weights and biases to Zero
    for i in model.conv1.children():
        torch.nn.init.normal_(i[0].weight, 0.0, 0.5)
        torch.nn.init.zeros_(i[0].bias)
    # set FC weights and biases
    torch.nn.init.normal_(model.fc.weight, 0.0, 0.5)
    torch.nn.init.zeros_(model.fc.bias)
    return model



# CNN2dx1 Cosine FC
########################################################################################################################
class Conv2d1CosFc1(nn.Module):

    def __init__(self, in_chans=1, out_chans=5, ks=([5, 6, 7, 3, 5, 6, 7, 3], 9), ks_avg=(1, 2)):
        super(Conv2d1CosFc1, self).__init__()

        self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=(ks[0][j], ks[1]), padding=(0, int((ks[1] - 1) / 2)), stride=(ks[0][j], 1)),
                                   nn.ReLU(),
                                   nn.AvgPool2d(kernel_size=ks_avg)) for j in range(8)])
        self.fc = nn.Linear(int((8 * 7) / 2), 2)

    def forward(self, x):
        n_chan_groups = [5, 6, 7, 3, 5, 6, 7, 3]
        indx_chan_groups = [range(sum(n_chan_groups[:k]), sum(n_chan_groups[:k + 1])) for k in range(len(n_chan_groups))]
        out1 = []
        # 42 Channels ->> 8 channel groups
        for j in range(8):
            out1.append(self.conv[j](x[:, :,  indx_chan_groups[j], :]))
        indx_rep = torch.arange(0, 5).unsqueeze(dim=1).repeat(1, 5).view(1, -1).squeeze()
        similarity = [cos(out1[j].squeeze().repeat(1, 5, 1), out1[k].squeeze()[:, indx_rep, :]).mean(dim=-1, keepdim=True) for j in range(8) for k in range(j+1, 8)]
        similarity = torch.cat(similarity, 1)
        out = similarity.view(similarity.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc(out)
        # out = torch.softmax(out, dim=1)  # classes are in columns
        return out#, similarity.mean(dim=0)


def weight_init_conv2d1cosfc1(model):
    # set CNN weights and biases to Zero
    for i in model.conv.children():
        torch.nn.init.normal_(i[0].weight, 0.0, 0.5)
        torch.nn.init.zeros_(i[0].bias)
    # set FC weights and biases
    torch.nn.init.normal_(model.fc.weight, 0.0, 0.5)
    torch.nn.init.zeros_(model.fc.bias)
    return model



# CNN2dx1 FCx2
########################################################################################################################
class CNN2d1_fc2(nn.Module):

    def __init__(self, in_chans=1, out_chans=5, ks=(42, 9), ks_avg=(1, 312), n_fc1=5):
        super(CNN2d1_fc2, self).__init__()

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=ks, stride=(42, 1), padding=(0, 0)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=ks_avg)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(out_chans, n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, 2)

    def forward(self, x):
        out = self.conv2(x)  # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def weight_init_cnn2d1_fc2(model):
    #import pdb; pdb.set_trace()
    # set CNN weights and biases to Zero
    torch.nn.init.normal_(model.conv2[0].weight, 0.0, 0.5)
    torch.nn.init.zeros_(model.conv2[0].bias)
    # set FC weights and biases
    torch.nn.init.normal_(model.fc1[0].weight, 0.0, 0.5)
    torch.nn.init.zeros_(model.fc1[0].bias)
    torch.nn.init.normal_(model.fc2.weight, 0.0, 0.5)
    torch.nn.init.zeros_(model.fc2.bias)
    return model


# EEGNet
########################################################################################################################
class EEGNet(nn.Module):
    def __init__(self, fs=64, T=64*5, F1=8, D=2, C=42, F2=None, dropout_rate=0.25):
        super(EEGNet, self).__init__()

        self.dropout_rate = dropout_rate

        # Block 1
        kernel_conv = (1, int(fs/2)-1)
        pad_conv = (0, int((kernel_conv[1]-1)/2)) # Since the kernel is one-dimensional, no need to pad along the 1 dimension
        in_channels = 1  # Here channel refers to depth NOT channels of EEG data

        # Here groups is specified to in_channels to implement depthwise convolution (no summation over channels)
        # However doesn't matter here as in_channels=1
        self.conv = nn.Conv2d(in_channels, F1, kernel_conv, padding=pad_conv, bias=False, groups=1)
        self.batchnorm1 = nn.BatchNorm2d(F1, affine=False)
        # Here once again a depthwise convolution (separate 2d kernels on each channel is applied)
        self.depthwise_conv1 = nn.Conv2d(in_channels=F1, out_channels=D*F1, kernel_size=(C, 1), padding=(0, 0), bias=False, groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(D*F1, affine=False)
        self.avgpooling1 = nn.AvgPool2d(1, 4)


        # Block 2
        k_conv3 = (1, int(fs/(2*4))-1)
        pad_conv3 = (0, int((k_conv3[1]-1)/2))

        if not F2:
            F2 = D * F1  # Page 6 paragraph 3 of paper

        self.depthwise_conv2 = nn.Conv2d(D*F1, D*F1, k_conv3, padding=pad_conv3, bias=False, groups=D*F1)
        # For pointwise covolution, groups is set to 1, in order to apply a 3d kernel, summing over all channels
        self.pointwise_conv = nn.Conv2d(D*F1, F2, kernel_size=(1, 1), padding=(0, 0), groups=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2, affine=False)
        self.avgpooling2 = nn.AvgPool2d(1, 8)

        # Fully Connected Layer
        self.fc = nn.Linear(F2*int(T//(4*8)), 1)

    def forward(self, x):

        dropout_rate = self.dropout_rate

        # Block 1
        x = self.conv(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv1(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.avgpooling1(x)
        x = F.dropout(x, dropout_rate)

        # Block 2
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.avgpooling2(x)
        x = F.dropout(x, dropout_rate)

        # Fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = torch.sigmoid(x).squeeze()
        return out


def normalize_weights_eegnet(model, norm_rate=4, eps=1e-8):
    # Double check gradients ...
    #print('wc applied!')
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'depthwise_conv1.weight' in name:
                param.copy_(param / (eps + param.norm(2, dim=(2, 3), keepdim=True)))
                #print(model.depthwise_conv1.weight.norm(2, dim=(2, 3)))
            if name == 'fc.weight' in name:
                param.copy_(param / (eps + norm_rate * param.norm(2, dim=1, keepdim=True)))
            #print(model.depthwise_conv1.weight.data[0, :, :, :].squeeze())







if __name__ == '__main__':

    # Test the model
    torch.manual_seed(123)
    #model = CnnCos1layer().to(torch.double)
    #model = CNN2d1_fc2(in_chans=1, out_chans=5).to(torch.double)
    #model = Cnn2d1CosFc1(in_chans=1, out_chans=5).to(torch.double)
    model = EEGNet().to(torch.double)
    Xb = torch.rand(128, 1, 42, 64*5).to(torch.double)
    out = model(Xb)
    #model1 = normalize_weights_eegnet(model)

    # model = CnnCos1layer()
    # summary(model, (42, 320))
    #summary(model, input_size, batch_size=-1, device='cuda')

    input1 = torch.randn(10, 10, 128)
    input2 = torch.randn(10, 100, 128)
    import torch.nn as nn
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    #output = cos(input1, input2)