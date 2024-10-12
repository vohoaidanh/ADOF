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
