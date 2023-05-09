apt-get update

fallocate -l 40G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/jammy/amdgpu-install_5.4.50400-1_all.deb
apt-get install ./amdgpu-install_5.4.50400-1_all.deb

amdgpu-install --usecase=rocm,hip,mllib --no-dkms

usermod -a -G video $LOGNAME
usermod -a -G render $LOGNAME

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

git clone https://github.com/pytorch/examples.git
cd examples/mnist
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python3 main.py

apt install rocrand 
apt install rocblas
apt install rocm-libs

docker pull rocm/pytorch:latest-base
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest-base

cd ~
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive

export PYTORCH_ROCM_ARCH=gfx1032

python3 tools/amd_build/build_amd.py
USE_ROCM=1 MAX_JOBS=12 python3 setup.py install --user

cd ..

git clone https://github.com/ThatOneShortGuy/Discord-AI-ChatBot.git
cd Discord-AI-ChatBot
pip install -r requirements.txt
