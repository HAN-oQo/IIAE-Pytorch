
"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

FILE=$1

if  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-v2-dataset" ]; then
    URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
    ZIP_FILE=./data/afhq_v2.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

else
    echo "Available arguments are pretrained-network-celeba-hq, pretrained-network-afhq, celeba-hq-dataset, and afhq-dataset."
    exit 1

fi