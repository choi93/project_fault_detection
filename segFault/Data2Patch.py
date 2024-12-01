import numpy as np
from segFault.data_normalization import PatchNormalization
from skimage.transform import resize
from segFault.utils.module_io import *
import yaml

class Data2patch():
    """
    Convert seismic data(Binary format) to training patches(npy format)

    매개변수는 config/config.yaml 파일에서 설정됩니다.
    """

    def __init__(self, config_path='config/data_preprocess_config.yaml'):
        # yaml 파일에서 설정 로드
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['data']
            
        # config에서 값 설정
        self.indir = config['indir']
        self.fname = self.indir + config['fname']
        self.lname = self.indir + config['lname']
        
        self.nmodel = config['nmodel']
        self.nx = config['nx']
        self.nz = config['nz']
        self.nwc = config['nwc']
        self.nzc = config['nzc']
        self.nxp = config['nxp']
        self.nzp = config['nzp']
        self.itype = config.get('itype', 3)  # itype의 기본값은 3
        
        # 출력 파일명도 config에서 가져오기
        self.seisname = config['seisname']
        self.labelname = config['labelname']

    def run(self):
        # seisname과 labelname을 매개변수로 받는 대신 클래스 속성 사용
        dpatch = self.data2patch()
        lpatch = self.label2patch()

        np.save(self.seisname, dpatch)
        np.save(self.labelname, lpatch)

    def data2patch(self):
    # convert seismic data to patch

        data = self.load_data()
        data_patch = self.make_patch(data)
        data_resize = self.patch_resize(data_patch)

        patch_norm = PatchNormalization(itype=self.itype)
        data_norm = patch_norm.run(data_resize)
 
        return data_norm

    def label2patch(self):
    # convert label data to patch
    
        data=self.load_data(self.lname,self.nx,self.nz,self.nwc,self.nzc)
        data_patch=self.make_patch(data,self.nxp,self.nzp)
    
        return data_patch
    
    def load_data(self):
    # load seismic data
        
        data=from_bin(self.fname,self.nx,self.nz)
        data_crop=np.zeros((self.nx,self.nzc-self.nwc))
        data_crop[:,:]=data[:,self.nwc:self.nzc]
        
        return data_crop
    
    
    def patch_resize(self, data_patch):
    # reshape를 사용하여 배열 재구성
        data_resize = np.array([resize(patch, (self.nxp, self.nzp), mode='constant', 
                                     preserve_range=True) for patch in data_patch])
        return data_resize
    
    def make_patch(self, data):
    #split data into patches
    
        nx, nz = data.shape
        xpatch = max(1, int(nx/self.nxp*2)-1)
        zpatch = int(nz/self.nzp*2)-1
        
        # 인덱스 배열 생성으로 루프 제거
        x_starts = np.arange(xpatch) * self.nxp//2
        z_starts = np.arange(zpatch) * self.nzp//2
        
        # meshgrid로 모든 시작점 조합 생성
        x_indices, z_indices = np.meshgrid(x_starts, z_starts)
        
        # 결과 배열 미리 할당
        patches = np.zeros((xpatch * zpatch, self.nxp, self.nzp))
        
        # 벡터화된 연산으로 패치 추출
        for i, (xs, zs) in enumerate(zip(x_indices.flat, z_indices.flat)):
            patches[i] = data[xs:xs+self.nxp, zs:zs+self.nzp]
        
        return patches