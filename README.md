# About
Adaptive Step Size Strategy for Novel Adversarial Example Generation
including negI-FGSM, RMS-negI-FGSM, Adam-negI-FGM, Nadam-negI-FGM.

# Usage
# Prerequisites
To run this code, ensure you have the following packages installed:
Ubuntu 22.04
CUDA 11.6
Python 3.6
Deep learning framework Tensorflow 1.15.2

Installation of these packages can be achieved through pip.

# Model Weights
Download the pre-trained weights for the victim models and CDMA from GoogleDrive. After downloading, unzip the file into the root directory of this project.

# Generation
python negI-FGSM.py --input_dir test_image --output_dir exp/negI-FGSM/epsilon7000_iter300/incepv4 --max_epsilon 7000 --num_iter 300  --image_width 299 --image_height 299 --batch_size 1 --momentum 0


python RMS-negI-FGSM.py --input_dir test_image --output_dir exp/RMS-negI-FGSM/epsilon7000_iter300/incepv4 --max_epsilon 7000 --num_iter 300  --image_width 299 --image_height 299 --batch_size 1 --momentum 0


python Adam-negI-FGM.py --input_dir test_image --output_dir exp/Adam-negI-FGM/epsilon7000_iter300/incepv4 --max_epsilon 7000 --num_iter 300  --image_width 299 --image_height 299 --batch_size 1 --momentum 0


python Nadam-negI-FGM.py --input_dir test_image --output_dir exp/Nadam-negI-FGM/epsilon7000_iter300/incepv4 --max_epsilon 7000 --num_iter 300  --image_width 299 --image_height 299 --batch_size 1 --momentum 0

# Test
python test.py --model_path models/inception_v4.ckpt --base_test_data_dir alpha/Nadam-negI-FGM --label_file dataset/labels.txt --subfolders your folders

