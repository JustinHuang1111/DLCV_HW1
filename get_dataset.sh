# Download dataset
gdown 1LMIaOY8NSKWmGtbvTsjXcHVaYKnNN_9u -O hw1_data.zip

# Unzip the downloaded zip file
mkdir hw1_data
unzip ./hw1_data.zip -d hw1_data

wget -O ./hw1a/ckpt/inceptionv3_best_0.87.ckpt https://www.dropbox.com/s/zeneltt0ht5mk7j/inceptionv3_best_0.87.ckpt?dl=0
wget -O ./hw1b/ckpt/DL_NLi_best.ckpt https://www.dropbox.com/s/1nanvauj918kzi4/DL_NLi_best.ckpt?dl=0
wget -O ./hw1b/ckpt/DL_N3_best.ckpt https://www.dropbox.com/s/uujpubfj1pufvf4/DL_N3_best.ckpt?dl=0
