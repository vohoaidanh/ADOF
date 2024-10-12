# Function to check if gdown is installed
sudo apt-get update -y
sudo apt-get install -y unzip -q
sudo apt-get install -y zip -q

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

#############################################
pwd=$(cd $(dirname $0); pwd)
echo pwd: $pwd

# pip install gdown==4.7.1 

mkdir dataset
cd dataset

# --proxy http://ip:port

# Download the zip file from the Google Drive folder using gdown
#https://drive.google.com/file/d/1JBDx2BPVSoADxhLFMJpd8zSFQyD_SVvI/view?usp=drive_link
gdown https://drive.google.com/uc?id=1JBDx2BPVSoADxhLFMJpd8zSFQyD_SVvI -O trainset.zip --continue

# Check if the download was successful
if [[ -f trainset.zip ]]; then
    echo "Download complete. Extracting file.zip..."
    
    # Unzip the downloaded file
    unzip trainset.zip -d ForenSynths_train

    # Check if unzip was successful
    if [[ $? -eq 0 ]]; then
        echo "Extraction complete. Deleting file.zip..."
        
        # Remove the zip file
        rm trainset.zip
        echo "file.zip has been deleted."
    else
        echo "Error during extraction."
    fi
else
    echo "Download failed."
fi

#Test set https://drive.google.com/file/d/1bvGFZ-77Xcu3sK7Pq9CyUBBk8PACzmeB/view?usp=drive_link

gdown https://drive.google.com/uc?id=1bvGFZ-77Xcu3sK7Pq9CyUBBk8PACzmeB -O testset.zip --continue

# Check if the download was successful
if [[ -f testset.zip ]]; then
    echo "Download complete. Extracting file.zip..."
    
    # Unzip the downloaded file
    mkdir -p ForenSynths_train/test
    unzip testset.zip -d ForenSynths_train/test

    # Check if unzip was successful
    if [[ $? -eq 0 ]]; then
        echo "Extraction complete. Deleting file.zip..."
        
        # Remove the zip file
        rm testset.zip
        echo "file.zip has been deleted."
    else
        echo "Error during extraction."
    fi
else
    echo "Download failed."
fi