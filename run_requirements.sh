#!/bin/bash

#./script.sh True


# Load repository
BRANCH_NAME="exp/pyramid"  # Replace with your branch name
REPO_URL="https://github.com/vohoaidanh/ADOF.git"  # Replace with your repository URL

# Clone the specific branch
git clone -b $BRANCH_NAME $REPO_URL


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

cd ADOF  # Change to the directory name of the cloned repository
pip install -r ./requirements.txt

#Load dataset
chmod +x ./download_trainset.sh
sed -i 's/\r//' ./download_trainset.sh

chmod +x run_dowload_testset.sh
sed -i 's/\r//' ./run_dowload_testset.sh

#Training
chmod +x ./run_train.sh
sed -i 's/\r//' ./run_train.sh
#./train.sh 'cnndetection' '0' '--use_comet' # backbone, gpu_ids, use_comet


