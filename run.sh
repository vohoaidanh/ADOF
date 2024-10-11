#!/bin/bash

#./script.sh True

download_dataset="${1:-False}"


# Load repository
BRANCH_NAME="exp/backbones"  # Replace with your branch name
REPO_URL="https://github.com/vohoaidanh/ADOF.git"  # Replace with your repository URL

# Clone the specific branch
git clone -b $BRANCH_NAME $REPO_URL
cd ADOF  # Change to the directory name of the cloned repository


#install packages
sudo apt-get update -y
sudo apt-get install -y libgl1-mesa-glx -q
check_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown not found. Installing..."
        pip install gdown
        echo "gdown has been installed successfully."
    else
        echo "gdown is already installed."
    fi
}
check_gdown
pip install -r requirements.txt

#Load dataset
if [[ "$download_dataset" == "True" ]]; then
    chmod +x download_trainset.sh
    sed -i 's/\r//' ./download_trainset.sh
    ./download_trainset.sh
else
    echo "Skipping dataset download."
fi


#Training
chmod +x train.sh
sed -i 's/\r//' ./train.sh
./train.sh 'cnndetection' '0' '--use_comet' # backbone, gpu_ids, use_comet


