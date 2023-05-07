sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/jammy/amdgpu-install_5.4.50400-1_all.deb
sudo apt-get install ./amdgpu-install_5.4.50400-1_all.deb

sudo amdgpu-install --usecase=rocm
sudo apt install python3-pip

cd Discord-AI-Chatbot

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
pip3 install -r requirements.txt