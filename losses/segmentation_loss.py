import torch
import torch.nn as nn

class SegmentationLossFunction:
    """
    사용자 정의 손실 함수
    - BCE 손실과 누적 손실을 결합
    - 순방향과 역방향 누적을 모두 고려
    """
    def __init__(self, device):
        self.device = device
        self.bce = nn.BCELoss().to(device)
    
    def smooth_step(self, x, k=100):
        return torch.sigmoid((x-0.5) * k)
    
    def cum_label_torch(self, label):
        label = self.smooth_step(label)
        diff = torch.relu(label[:, :, 1:, :] - label[:, :, :-1, :])
        
        return torch.cumsum(diff, dim=2)
        
    
    def cum_loss(self, pred, label, norm='L2', direction='forward', normalization=None):
        if direction == 'reverse':
            pred = torch.flip(pred, dims=[2])
            label = torch.flip(label, dims=[2])
        
        label = self.cum_label_torch(label)
        pred = self.cum_label_torch(pred)
        
        if norm == 'L2' and normalization==None:
            loss = torch.mean((pred - label)**2)
        elif norm == 'L2' and normalization==True:
            loss = torch.mean(((pred - label)/(label+1))**2)
        elif norm == 'L1' and normalization==None:
            loss = torch.mean(torch.abs(pred - label))
        elif norm == 'L1' and normalization==True:
            loss = torch.mean(torch.abs((pred - label)/(label+1)))
        
        return loss
    
def __call__(self, predictions, labels, k, use_seg_loss=True, norm='L2', normalization=None):
        if not use_seg_loss:
            return self.bce(predictions, labels)

        loss1 = self.bce(predictions, labels)
        loss2 = self.cum_loss(predictions, labels, norm, 'forward', normalization)
        loss3 = self.cum_loss(predictions, labels, norm, 'reverse', normalization)
        
        return loss1*k[0] + loss2*k[1] + loss3*k[2] 
