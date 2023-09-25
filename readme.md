# Exploring Bias in StyleCLIP (StyleGAN + CLIP)








Setup ilabu Project:

Open terminal , check by the below 2 cmds, that it has cuda and ubunutu.

=> nvidia-smi

Cuda version 11.4

=> lsb_release -a

Ubuntu version 20.04 LTS

Anaconda path:
export PATH="$PATH:/koko/system/anaconda/bin"

source activate python38

pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

Download project from github.
cd StyleCLIP
Create new folder pretrained_models inside StyleCLIP folder.
Download model_ir_se50.pth to /pretrained_models 

Download stylegan2-ffhq-config-f.pt to /pretrained_models		-- wget https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT -O stylegan2-ffhq-config-f.pt

keep-job 80
export PATH="$PATH:/koko/system/anaconda/bin"
source activate python38

Editing via Latent Mapper:

cd mapper
Try first this: python3 scripts/train.py --exp_dir ../results/surprised --no_fine_mapper --description "surprised" --latents_train_path ../../e4e_test_results/latents.pt --train_dataset_size 10 --batch_size 2


Testing:

From project's root:
python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path results/mohawk_hairstyle/checkpoints/best_model.pt --couple_outputs --n_images 1 --no_fine_mapper

python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path results/mohawk_hairstyle/checkpoints/best_model.pt --couple_outputs --n_images 2 --no_fine_mapper --latents_test_path mapper/test_faces.pt 

Download surprised.pt : https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI, at pretrained_models inside StyleCLIP folder.

python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path pretrained_models/surprised.pt --couple_outputs --n_images 5 --no_fine_mapper --latents_test_path mapper/test_faces.pt 



-- Source 2:
2nd Method, training Latent Mapper:

This Latent Mapper is text specific. Takes ~12 hours for each text.
This latent mapper part of the architecture is to be trained.

stylegan2-ffhq-config-f.pt => Pretrained StyleGAN model and weights
model_ir_se50.pt => Pretrained FaceRecognition model and weights => ArcFace model for identifying whether 2 images are of same person or different person.

Inverted (by e4e: Encoder for editing), train and test Celeb-A-HQ dataset is provided. (Latent vectors , w are provided)

StyleGAN and FaceRecognition models are pretrained. No need to retrain it.

We need to train latent mapper AND, for that, Clip loss, ID loss and L2 loss are required.

For training, we need to have inverted train data (train w), which is train_faces.pt.
For testing, we also need to have inverted test data (test w), which is test_faces.pt.


Setup ilabu Project:


=> nvidia-smi

Cuda version 11.4

=> lsb_release -a

Ubuntu version 20.04 LTS

Anaconda path:
export PATH="$PATH:/koko/system/anaconda/bin"

source activate python38

pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

Download model_ir_se50.pth to /pretrained_models 

Download stylegan2-ffhq-config-f.pt to /pretrained_models		-- wget https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT -O stylegan2-ffhq-config-f.pt

Download test_faces.pt to /
Download train_faces.pt to /

Editing via Latent Mapper:
cd mapper
python3 scripts/train.py --exp_dir ../results/mohawk_hairstyle --no_fine_mapper --description "mohawk hairstyle"

From project's root:
python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path results/mohawk_hairstyle/checkpoints/best_model.pt --couple_outputs --n_images 1 --no_fine_mapper

python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path results/mohawk_hairstyle/checkpoints/best_model.pt --couple_outputs --n_images 2 --no_fine_mapper --latents_test_path mapper/test_faces.pt 

Download surprised.pt : https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI

python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path pretrained_models/surprised.pt --couple_outputs --n_images 5 --no_fine_mapper --latents_test_path mapper/test_faces.pt 

----------
For e4e_env:

Setting up conda env:

export PATH="$PATH:/koko/system/anaconda/bin"

source activate python36

change the torchvision==0.7.0 in environment/env.yaml

launch a new env using some conda cmd reading from the above yaml file.

conda activate e4e_env

export PYTHONPATH="$PYTHONPATH:~Desktop/MLProject/encoder4editing"

run e4e_model




python3 scripts/train.py --exp_dir ../results/surprised --no_fine_mapper --description "surprised" --latents_train_path ../../e4e_test_results/latents.pt --train_dataset_size 10 --batch_size 2

testing:
python3 mapper/scripts/inference.py --exp_dir ../../test_results/mohawk_hairstyle --checkpoint_path pretrained_models/surprised.pt --couple_outputs --n_images 5 --no_fine_mapper --latents_test_path mapper/test_faces.pt 

srun -G 1 --pty python3 mapper/scripts/inference.py --exp_dir ../final_test_results/surprised_on_utk/ --checkpoint_path results/surprisedNew/checkpoints/best_model.pt --couple_outputs --n_images 40 --no_fine_mapper --latents_test_path ../e4e_test_results/latents.pt

continuing from checkpoint:
python3 scripts/train.py --exp_dir ../results/surprisedNew --no_fine_mapper --description "surprised" --latents_train_path ../../e4e_test_results/latents.pt --train_dataset_size 10 --batch_size 2
--checkpoint_path ../results/surprised/checkpoints/iteration_14000.pt



-- Using Encoder4Editing:
git clone https://github.com/omertov/encoder4editing.git
cd encoder4editing

Raw dataset path: /common/home/kv353/freespace/local/kv353/StyleClipProject/dataset_raw/dataset_01





-- Mine:
export PYTHONPATH="$PYTHONPATH:/common/home/kv353/freespace/local/kv353/StyleCLIP-main"
srun -G 1 --pty python3 scripts/inference.py --exp_dir ../../test_results/TaylorSwift --checkpoint_path ../results/TaylorSwift/checkpoints/best_model.pt --couple_outputs --n_images 40 --no_fine_mapper --latents_test_path test_faces.pt
srun -G 1 --pty python3 scripts/inference.py --exp_dir ../../test_results/TaylorSwift_From_latents.pt --checkpoint_path ../results/TaylorSwift/checkpoints/best_model.pt --couple_outputs --n_images 40 --no_fine_mapper --latents_test_path ../../e4e_test_results/latents.pt

-- New:
export PYTHONPATH="$PYTHONPATH:/common/home/kv353/freespace/local/kv353/StyleClipProject/StyleCLIP-main"
srun -G 1 --pty python3 mapper/scripts/inference.py --exp_dir ../test_results/TaylorSwift --checkpoint_path results/TaylorSwift/checkpoints/best_model.pt --couple_outputs --n_images 40 --no_fine_mapper --latents_test_path mapper/test_faces
srun -G 1 --pty python3 mapper/scripts/inference.py --exp_dir ../test_results/TaylorSwift_From_latents.pt --checkpoint_path results/TaylorSwift/checkpoints/best_model.pt --couple_outputs --n_images 40 --no_fine_mapper --latents_test_path ../e4e_test_results/latents.pt
