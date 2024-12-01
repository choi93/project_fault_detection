import numpy as np
from sklearn.preprocessing import QuantileTransformer

class PatchNormalization():
    """
    패치 정규화 수행을 위한 클래스

    매개변수
    itype: 1-5 까지의 정수로 정규화 기법 결정을 위해 사용(default = 3)
    itype=1 -> max scaling (절대값의 최대값으로 스케일링)
    itype=2 -> min-max scaling (0~1 사이 범위로 스케이링, standardization)
    itype=3 -> gaussian scaling (표준 정규 분포 형태로 스케일링, normalization)
    itype=4 -> symmetric cutting (양수와 음수 범위가 일치하도록 자료 커팅)
    itype=5 -> normal score transform (자료의 분포가 정규분포 형태로 변환)

    사용법
    patch_norm=PatchNormalization(itype=itype)
    data_nrom = patch_norm.run(data)
    data_nrom = patch_norm.run(data,seed) (itype=5 인 경우에만 seed 값 필요)

    """

    def __init__(self,itype=3):

        self.itype=itype

    def run(self,data,seed=0):

        if self.itype==1:
            data_norm=self.max_norm(data)
        elif self.itype==2:
            data_norm=self.minmax_norm(data)
        elif self.itype==3:
            data_norm=self.gaussian_norm(data)
        elif self.itype==4:
            data_norm=self.symetric_cutting(data)
        elif self.itype==5:
            data_norm=self.normal_score_transform(data,seed)

        return data_norm

    def max_norm(self,data):
        data_norm=data/np.abs(data).max()

        return data_norm

    def minmax_norm(self,data):
        dmax=np.max(data)
        dmin=np.min(data)
        data_norm = (data-dmin)/(dmax-dmin)
        
        return data_norm

    def gaussian_norm(self,data):

        dm=np.mean(data)
        ds=np.std(data)
        data_norm=(data-dm)/ds

        return data_norm

    def symetric_cutting(self,data):
        data_sort = np.sort(data.ravel())
        b1 = np.abs(data_sort[0])
        b2 = np.abs(data_sort[-1])
        
        if b1 > b2:
            data = np.clip(data, -b2, None)
            data_norm = data/b2
        else:
            data = np.clip(data, None, b1)
            data_norm = data/b1
        
        return data_norm

    def normal_score_transform(self,data,seed=0):
        transformer = QuantileTransformer(output_distribution='normal',random_state=seed)
        dtmp = data.reshape(-1,1).copy()
        transformed_dtmp = transformer.fit_transform(dtmp)
        data_norm = transformed_dtmp.reshape(data.shape)

        return data_norm