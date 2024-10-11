#!/bin/bash

# Load repository
BRANCH_NAME="exp/backbones"  # Replace with your branch name
REPO_URL="https://github.com/vohoaidanh/ADOF.git"  # Replace with your repository URL

# Clone the specific branch
git clone -b $BRANCH_NAME $REPO_URL
cd ADOF  # Change to the directory name of the cloned repository

#Load dataset
chmod +x download_trainset.sh
./download_trainset.sh


#Training
chmod +x train.sh
./train.sh
