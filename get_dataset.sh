# # Download dataset
gdown 1LMIaOY8NSKWmGtbvTsjXcHVaYKnNN_9u -O hw1_data.zip

# # Unzip the downloaded zip file
mkdir hw1_data
unzip ./hw1_data.zip -d hw1_data

wget -O model1.ckpt 'https://www.dropbox.com/s/a0o8feccpugtwix/model1.ckpt?dl=1'
wget -O model2.ckpt 'https://www.dropbox.com/s/0ty3tsy2sa0jfgv/model2.ckpt?dl=1'
