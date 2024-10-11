# Function to check if gdown is installed
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
    unzip trainset.zip -d temp_dir
    mv temp_dir/* ./ForenSynths/
    rmdir temp_dir

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

gdown https://drive.google.com/uc?id=1JBDx2BPVSoADxhLFMJpd8zSFQyD_SVvI -O testset.zip --continue

# Check if the download was successful
if [[ -f testset.zip ]]; then
    echo "Download complete. Extracting file.zip..."
    
    # Unzip the downloaded file
    unzip testset.zip -d temp_dir
    mv temp_dir/* ./ForenSynths/test
    rmdir temp_dir

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

 ==============================================================================
 
 # https://github.com/Yuheng-Li/UniversalFakeDetect
 # https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-
 gdown https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO- -O ./UniversalFakeDetect --folder
 cd ./UniversalFakeDetect
 ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
 cd $pwd/dataset
 
 # https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
 # https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing
 gdown https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj -O ./GANGen-Detection --folder
 
 cd ./GANGen-Detection
 ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
 cd $pwd/dataset
 
 # https://github.com/ZhendongWang6/DIRE
 # https://drive.google.com/drive/folders/1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD?usp=sharing
 gdown https://drive.google.com/drive/folders/1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD -O ./DiffusionForensics --folder
 
 cd ./DiffusionForensics
 ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
 cd $pwd/dataset

 ==============================================================================
