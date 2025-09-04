# 로그 변환 : 편차가 큰 데이터를 로그변환하면 분포를 개선하고, 큰 범위 차이를 줄이며,
# 모델이 보다 안정적으로 학습할 수 있도록 만들어 주는 장점이 있음
import numpy as np
np.set_printoptions(suppress=True, precision=6) # precision -> 소수점 몇번째 자리까지 추출 
def test():
    values = np.array([3.45,34.5,0.345,0.01, 0.1, 10, 100,1000])
    print(np.log2(3.45), np.log10(3.45), np.log(3.45))
    print('원본 자료:', values)
    log_values = np.log10(values) # 상용로그 
    print('log_values: ', log_values)
    ln_values = np.log(values) # 자연로그 
    print('ln_values: ', ln_values)

    # 표준화 : 값을 평균을 기준으로 분포 시켜 음수부터 양수까지 데이터 분포가 나올 수 있음
    # 정규화 : 정규화는 데이터의 범위를 0 ~ 1 사이로 변환해 데이터 분포를 조정
    # 로그값의 최소, 최대를 구해 0 ~ 1 사이 범위로 정규화
    min_log = np.min(log_values)
    max_log = np.max(log_values)
    normalized = (log_values - min_log) / (max_log - min_log)
    print('정규화 결과: ', normalized)

def log_inverse():
    offset = 1 
    log_value = np.log(10 + offset)
    print('log_value: ', log_value)
    original = np.exp(log_value) - offset # np.exp() 로그 변환에 역변환 가능 
    print('original: ', original)

class LogTrans: 
    def __init__(self, offset:float=1.0):
        self.offset = offset

    # 로그변환 수행 메소드 
    def transform(self, x:np.ndarray):
        # fx() = log(x + offset)
        return np.log(x + self.offset)  # 자연로그를 취함 
    
    # 역변환 수행 메소드 
    def inverse_trans(self, x_log:np.ndarray):
        return np.exp(x_log) - self.offset
    
def gogo(): 
    print('~' * 20)
    data = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], dtype=float)
    # 로그 변환용 클래스 개체 생성
    log_trans = LogTrans(offset=1.0)

    # 로그 변환 및 역변환
    data_log_scaled = log_trans.transform(data)
    recover_data = log_trans.inverse_trans(data_log_scaled)
    print('원본 데이터:', data)
    print('로그변환된  데이터:', data_log_scaled)
    print('역변환된 원본 데이터:', recover_data)

if __name__ == "__main__":
    test()
    log_inverse()
    gogo()