import torch
import torch.optim as optim
from models.unet import UNet
from data.data_module import DataModule
from losses.segmentation_loss import SegmentationLossFunction
import os
import time
from tqdm import tqdm

class Trainer:
    """
    모델 학습을 관리하는 클래스
    """
    def __init__(self, model, optimizer, loss_fn, device, save_dir='./models'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
    def train_step(self, inputs, labels, k, epoch, use_seg_loss=True, norm='L2', normalization=None):
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
    
    def train(self, train_loader, val_loader, num_epochs, k, use_seg_loss=True, norm='L2', normalization=None):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            epoch_start_time = time.time()
            
            self.model.train()
            epoch_loss = 0.0
            
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                loss = self.train_step(inputs, labels, k, epoch, use_seg_loss, norm, normalization)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(train_loader)
            val_loss = self.validate(val_loader, k, use_seg_loss, norm, normalization)
            
            self._save_checkpoint(epoch, avg_loss, val_loss)
            self._log_epoch_results(epoch, avg_loss, val_loss, epoch_start_time)
    
    def _save_checkpoint(self, epoch, train_loss, val_loss):
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

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = 100
    epochs = 600
    k = [1., 1., 1.]

    # Setup data
    data_module = DataModule('./', batch_size=batch_size, device=device)
    data_module.setup(
        train_file=('s_train.npy', 'e_train.npy'),
        val_file=('s_val.npy', 'e_val.npy')
    )
    
    # Initialize model and training components
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = SegmentationLossFunction(device)
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Create trainer and start training
    trainer = Trainer(model, optimizer, loss_fn, device)
    trainer.train(
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        num_epochs=epochs,
        k=k,
        use_seg_loss=True,
        norm='L2',
        normalization=None
    ) 
