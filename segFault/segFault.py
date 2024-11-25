import torch
import torch.optim as optim
from segFault.unet import UNet
from segFault.utils.data_module import DataModule
from segFault.segmentation_loss import SegmentationLossFunction
import os
import time
from tqdm import tqdm
import yaml  # YAML 파싱을 위해 추가
from omegaconf import OmegaConf  # 선택적: 더 복잡한 설정 관리 시 유용

class Trainer:
    """
    모델 학습을 관리하는 클래스
    """
    def __init__(self, config_path='input_train.yaml'):
        # 설정 불러오기
        self.params = self.load_config(config_path)
        
        self.device = torch.device(self.params['device'])
        torch.backends.cudnn.benchmark = True
        
        # 데이터 설정
        self.data_module = DataModule(
            self.params['data_dir'],
            batch_size=self.params['batch_size'],
            device=self.device
        )
        self.data_module.setup(
            train_file=self.params['train_file'],
            val_file=self.params['val_file']
        )
        
        # 모델 및 학습 구성 요소 초기화
        self.model = UNet().to(self.device)
        if self.params['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        self.loss_fn = SegmentationLossFunction(self.device)
        
        self.save_dir = './models'
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_loss = float('inf')
        self.best_epoch = 0

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def train_step(self, inputs, labels, k, use_seg_loss=True, norm='L2', normalization=None):
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, labels, k, use_seg_loss, norm, normalization)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validate(self, val_loader, k, use_seg_loss=True, norm='L2', normalization=None):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels, k, use_seg_loss, norm, normalization)
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def train(self):
        for epoch in range(self.params['epochs']):
            print(f"Epoch {epoch+1}/{self.params['epochs']}")
            epoch_start_time = time.time()
            
            self.model.train()
            epoch_loss = 0.0
            
            for inputs, labels in tqdm(self.data_module.train_loader, desc=f"Training Epoch {epoch+1}"):
                loss = self.train_step(inputs, labels, self.params['k'], self.params['use_seg_loss'], self.params['norm'], self.params['normalization'])
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(self.data_module.train_loader)
            val_loss = self.validate(self.data_module.val_loader, self.params['k'], self.params['use_seg_loss'], self.params['norm'], self.params['normalization'])
            
            self._save_checkpoint(epoch, val_loss)
            self._log_epoch_results(epoch, avg_loss, val_loss, epoch_start_time)
    
    def _save_checkpoint(self, epoch, val_loss):
        torch.save(self.model.state_dict(), f'{self.save_dir}/model{epoch}.pth')
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
    
    def _log_epoch_results(self, epoch, train_loss, val_loss, start_time):
        duration = time.time() - start_time
        with open("val_loss.txt", 'a') as fv:
            fv.write(f"{epoch}, {train_loss}, {val_loss}\n")
        
        print(f"Epoch {epoch+1} completed in {duration//60:.0f}m {duration%60:.0f}s")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Best model epoch: {self.best_epoch}") 

