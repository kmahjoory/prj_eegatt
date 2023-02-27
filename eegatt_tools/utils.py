import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as T


def tensor_info(tensor, text=''):   
  print(f'{text}  SHAPE:{list(tensor.shape)}  DEVICE:{tensor.device}  RANGE:{[tensor.min().item(), tensor.max().item()]} ')
 

def dl_info(dl):
  print(f'N-Batches:{len(list(dl))}  Batch_shape_X:{list(list(dl)[0][0].shape)}  Batch_shape_Y:{list(list(dl)[0][1].shape)}')


def create_img_dataloader(image_folder, transform=None, batch_size=25, shuffle=False, num_workers=2):
    if transform is None:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    img_dataset = datasets.ImageFolder(image_folder, transform)
    img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size, shuffle, num_workers)
    return img_dataset, img_dataloader