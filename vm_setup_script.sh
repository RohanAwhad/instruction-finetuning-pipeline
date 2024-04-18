echo "========= Installing Python 3.9 ========"
sudo apt-get update
sudo apt-get upgrade
sudo apt -y install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y install python3.9
sudo apt -y install python3.9-dev python3-pip python3.9-venv

echo "========= Installing CUDA ========"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11.7
