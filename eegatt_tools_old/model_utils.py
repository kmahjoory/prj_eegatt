import torch
import torch.nn as nn



# 1-CNN1D 1-FC
##############################

class CNN1d1_fc1(nn.Module):
    
    def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1=3, n_ch1=64, n_fc1=100):
        super(CNN1d1_fc1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1-1)/2)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int(n_inpd2/2*n_ch1), n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, n_outs)
        
    def forward(self, x):
        out = self.conv1(x)   # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
        out = self.fc1(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1) # classes are in columns
        return out


# 1-CNN1D 2-FC
##############################

class CNN1d1_fc2(nn.Module):
    
    def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1=3, n_ch1=64, n_fc1=100, n_fc2=100, ks_avg1=2):
        super(CNN1d1_fc2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1-1)/2)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=ks_avg1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int(n_inpd2/ks_avg1*n_ch1), n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_fc1, n_fc2),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(n_fc2, n_outs)
        
    def forward(self, x):
        out = self.conv1(x)   # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.softmax(out, dim=1) # classes are in columns
        return out


# 2-CNN1D 1-FC
##############################

class CNN1d2_fc1(nn.Module):
    
    def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1=3, ks2=3, n_ch1=64, n_ch2=64, n_fc1=100, ks_avg1=1, ks_avg2=2):
        super(CNN1d2_fc1, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1-1)/2)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=ks_avg1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_ch1, n_ch2, kernel_size=ks2, padding=int((ks2-1)/2)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=ks_avg2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int(n_inpd2/(ks_avg1*ks_avg2) * n_ch1), n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, n_outs)
        
    def forward(self, x):
        out = self.conv1(x)   # (bs, C, H,  W)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
        out = self.fc1(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1) # classes are in columns
        return out



# CNN1D 2-FC
##############################

class CNN1d2_fc2(nn.Module):
    
    def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1=3, ks2=3, n_ch1=64, n_ch2=64, n_fc1=100, n_fc2=100, ks_avg1=1, ks_avg2=2):
        super(CNN1d2_fc2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1-1)/2)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=ks_avg1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_ch1, n_ch2, kernel_size=ks2, padding=int((ks2-1)/2)),
            nn.Dropout(p=.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=ks_avg2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int(n_inpd2/(ks_avg1*ks_avg2) * n_ch1), n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_fc1, n_fc2),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(n_fc2, n_outs)
        
    def forward(self, x):
        out = self.conv1(x)   # (bs, C, H,  W)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.softmax(out, dim=1) # classes are in columns
        return out



# CNN1D-1 2-head 1-fc
########################
class CNN1d1_h2_fc1(nn.Module):
  
  def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1_1=3, ks1_2=5, n_ch1_1=64, n_ch1_2=64, n_fc1=100, ks_avg1_1=2, ks_avg1_2=2):
      super(CNN1d1_h2_fc1, self).__init__()
      
      self.conv1_1 = nn.Sequential(
          nn.Conv1d(n_inpch, n_ch1_1, kernel_size=ks1_1, padding=int((ks1_1-1)/2)),
          nn.Dropout(p=.5),
          nn.ReLU(),
          nn.AvgPool1d(kernel_size=2)#640-4-4)
      )
      self.conv1_2 = nn.Sequential(
          nn.Conv1d(n_inpch, n_ch1_2, kernel_size=ks1_2, padding=int((ks1_2-1)/2)),
          nn.Dropout(p=.5),
          nn.ReLU(),
          nn.AvgPool1d(kernel_size=2)#640-4-4)
      )     
      self.fc1 = nn.Sequential(
          nn.Linear(int(n_inpd2/2*(n_ch1_1+ n_ch1_2)), n_fc1),
          nn.ReLU()
      )
      self.fc2 = nn.Linear(n_fc1, n_outs)
      # 224 was the input image size. After twice max-pooling becomes 224/4
      # Number of channels over layers: 3 -> 16 -> 32
      
  def forward(self, x):
      out = torch.cat((self.conv1_1(x), self.conv1_2(x)), 2)   # (bs, C, H,  W)
      #out = torch.cat((out, self.conv1_19(x)), 2)
      #out = self.conv2_7(out)
      # Think about this transition
      out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
      out = self.fc1(out)
      #import pdb; pdb.set_trace()
      out = self.fc2(out)
      #import pdb; pdb.set_trace()
      out = torch.softmax(out, dim=1) # classes are in columns
      
      #import pdb; pdb.set_trace()
      return out



# 1-CNN1D 2-head 2-fc
########################
class CNN1d1_h2_fc2(nn.Module):
  
  def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1_1=3, ks1_2=5, n_ch1_1=64, n_ch1_2=64, n_fc1=100, n_fc2=100, ks_avg1_1=2, ks_avg1_2=2):
      super(CNN1d1_h2_fc2, self).__init__()
      
      self.conv1_1 = nn.Sequential(
          nn.Conv1d(n_inpch, n_ch1_1, kernel_size=ks1_1, padding=int((ks1_1-1)/2)),
          nn.Dropout(p=.5),
          nn.ReLU(),
          nn.AvgPool1d(kernel_size=ks_avg1_1)#640-4-4)
      )
      self.conv1_2 = nn.Sequential(
          nn.Conv1d(n_inpch, n_ch1_2, kernel_size=ks1_2, padding=int((ks1_2-1)/2)),
          nn.Dropout(p=.5),
          nn.ReLU(),
          nn.AvgPool1d(kernel_size=ks_avg1_1)#640-4-4)
      )     
      self.fc1 = nn.Sequential(
          nn.Linear(int(n_inpd2/ks_avg1_1*(n_ch1_1+ n_ch1_2)), n_fc1), # Here heads are concatenated along the channels, so ks_avg1_1 = ks_avg1_2
          nn.ReLU()
      )
      self.fc2 = nn.Sequential(
          nn.Linear(n_fc1, n_fc2),
          nn.ReLU()
        )
      self.fc3 = nn.Linear(n_fc2, n_outs)
      # 224 was the input image size. After twice max-pooling becomes 224/4
      # Number of channels over layers: 3 -> 16 -> 32
      
  def forward(self, x):
      out = torch.cat((self.conv1_1(x), self.conv1_2(x)), 2)   # (bs, C, H,  W)
      #out = torch.cat((out, self.conv1_19(x)), 2)
      #out = self.conv2_7(out)
      # Think about this transition
      out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
      out = self.fc1(out)
      #import pdb; pdb.set_trace()
      out = self.fc2(out)
      out = self.fc3(out)
      #import pdb; pdb.set_trace()
      out = torch.softmax(out, dim=1) # classes are in columns
      
      #import pdb; pdb.set_trace()
      return out



# 2-CNN1D batch 1-fc
########################

class CNN1d2_batch2_fc1(nn.Module):
    
  def __init__(self, n_inpch, n_inpd2, n_outs=2, ks1=3, ks2=3, n_ch1=64, n_ch2=64, n_fc1=100, ks_avg1=2, ks_avg2=2, st_avg1=2, st_avg2=2):
      super(CNN1d2_batch2_fc1, self).__init__()
      
      self.conv1 = nn.Sequential(
          nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1-1)/2)),
          nn.BatchNorm1d(n_ch1),
          nn.ReLU(),
          nn.AvgPool1d(kernel_size=ks_avg1, stride=st_avg1)
      )
      self.conv2 = nn.Sequential(
          nn.Conv1d(n_ch1, n_ch2, kernel_size=ks2, padding=int((ks2-1)/2)),
          nn.BatchNorm1d(n_ch2),
          nn.ReLU(),
          nn.AvgPool1d(kernel_size=ks_avg2, stride=st_avg2)
      )
      self.fc1 = nn.Sequential(
          nn.Linear(int(n_inpd2/(st_avg1+st_avg2) * n_ch2), n_fc1),
          nn.ReLU()
      )
      self.fc2 = nn.Linear(n_fc1, n_outs)
      
  def forward(self, x):
      out = self.conv1(x)   # (bs, C, H,  W)
      out = self.conv2(out)
      out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy 
      out = self.fc1(out)
      #out = torch.sigmoid(out)
      out = self.fc2(out)
      #out = torch.softmax(out, dim=1) # classes are in columns
      return out




