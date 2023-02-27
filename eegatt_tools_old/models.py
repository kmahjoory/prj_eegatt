import torch
import torch.nn as nn
import torch.optim as optim

cos = nn.CosineSimilarity(dim=1)


# In this model, EEG channels are split into 8 groups, first 4 corresponding to the one hemisphere & and the next 4 to the
# other hemisphere.
# In the first CNN layer, a 1dconv is applied for each group of channels separately! to extract temporal structure.
# In the 2nd CNN layer, a 1dconv is applied for each pair of channels resulting in 8*8/2
class CNN_k1(nn.Module):

    def __init__(self, n_inpch=[5, 6, 7, 3, 5, 6, 7, 3], n_inpd2=5*64, n_outs=2, ks1=9, ks2=3, n_ch1=1, n_fc1=8):
        super(CNN_k1, self).__init__()
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
        return out, similarity.mean(dim=0)


if __name__ == '__main__':

    # Test the model
    torch.manual_seed(123)
    model = CNN_k1().to(torch.double)
    Xb = torch.rand(2, 42, 64*5).to(torch.double)
    out, similarity = model(Xb)