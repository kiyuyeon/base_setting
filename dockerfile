# 1. CUDA 12.5 + cuDNN9 + Ubuntu 22.04 베이스
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Python, pip, git 설치
RUN apt-get clean && rm -rf /var/lib/apt/lists/* \
    && apt-get update -o Acquire::CompressionTypes::Order::=gz \
    && apt-get install -y python3 python3-pip git


# pip 업그레이드
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3. 파이썬 업그레이드 및 필수 패키지
RUN pip3 install --upgrade pip setuptools wheel

# 4. TensorFlow GPU (2.20.0)
RUN pip3 install tensorflow==2.20.0

# 5. C3AE 저장소 클론
WORKDIR /workspace
RUN git clone https://github.com/StevenBanama/C3AE.git
WORKDIR /workspace/C3AE

# 6. 요구 패키지 설치
RUN pip3 install -r requirements2.1.txt

# 7. 기본 실행 명령 (필요시 수정 가능)
#   -gpu: GPU 사용
#   -se : 사전 학습 모델 평가
#   --source imdb: 데이터셋 소스
CMD ["python3", "nets/C3AE.py", "-gpu", "-se", "--source", "imdb"]
