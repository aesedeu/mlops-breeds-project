#!/bin/sh
source .venv/bin/activate
gdown "https://drive.google.com/uc?id=1SyrHRYp3IRZ7b-vPOKJNJWn7hj97gFWa" -O breeds.zip
unzip breeds.zip -d data_temp
mkdir data
mv data_temp/data/* data/
rm -rf data_temp
rm breeds.zip
