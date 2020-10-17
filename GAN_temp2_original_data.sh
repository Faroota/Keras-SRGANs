#$ -l tmem=18G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /home/faljishi/logGAN
source /share/apps/source_files/cuda/cuda-9.0.source


cd /home/faljishi/Keras-SRGANs-original

#conda create -n GAN python=3.6.8
source activate GAN3
#source activate srgan

# for training:
# python train.py --input_dir='./data/' --output_dir='./output/' --model_save_dir='./model/' --batch_size=64 --epochs=3000 --number_of_images=1000 --train_test_ratio=0.8

#python train.py -i='./data/' -o='./output/' -m='./model/' -b=64 -e=2 -n=500 -r=0.8
# default
python train.py -h

# for testing:
#python test.py --input_high_res='./data_hr/' --output_dir='./output/' --model_dir='./model/gen_model3000.h5' --number_of_images=25 --test_type='test_model'

python test.py -h

