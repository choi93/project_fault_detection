import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class DataModule:
    """
    데이터 로딩과 처리를 담당하는 모듈
    - 학습 및 검증 데이터셋 로드
    - DataLoader 생성
    """
    def __init__(self, data_dir, batch_size, device):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        
    def load_data(self, fname):
        """
        데이터 로드
        """
        data = np.load(self.data_dir + fname).astype(np.float32)
        data = np.expand_dims(data, axis=1)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def setup(self, train_file=None, val_file=None, val_ratio=None, seed=42):
        """
        데이터 로드 및 데이터로더 생성
        """
        if train_file:
            s_train = self.load_data(train_file[0])
            e_train = self.load_data(train_file[1])
            
            if val_file is None and val_ratio is not None:
                if not 0 < val_ratio < 1:
                    raise ValueError("val_ratio는 0과 1 사이의 값이어야 합니다.")
                
                s_train, s_val, e_train, e_val = train_test_split(
                    s_train, e_train, 
                    test_size=val_ratio, 
                    random_state=seed
                )
                
                val_dataset = TensorDataset(s_val, e_val)
                self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            train_dataset = TensorDataset(s_train, e_train)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if val_file:
            s_val = self.load_data(val_file[0])
            e_val = self.load_data(val_file[1])
            val_dataset = TensorDataset(s_val, e_val)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size) 