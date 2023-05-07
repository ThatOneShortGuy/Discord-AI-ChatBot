sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/jammy/amdgpu-install_5.4.50400-1_all.deb
sudo apt-get install ./amdgpu-install_5.4.50400-1_all.deb

sudo amdgpu-install --usecase=rocm

cd Discord-AI-Chatbot
