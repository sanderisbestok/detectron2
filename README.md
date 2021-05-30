:warning: **This is not the original Detectron2**: This version is edited so it can be used for my Thesis. For the official Detectron2 repo please go to [this repository](https://github.com/facebookresearch/detectron2).

## Description
This readme will provide, installation, execution and testing instructions. Experiments for the thesis are conducted on the Lisa cluster and all installations are done in an Anaconda Environment.

Algorithms used for the thesis are:

 1. [ExtremeNet](https://github.com/sanderisbestok/ExtremeNet)
 2. [TridentNet in Detectron2](https://github.com/sanderisbestok/detectron2)
 3. [YoloV5](https://github.com/sanderisbestok/yolov5)

Evaluation tools can be found in [this repository](https://github.com/sanderisbestok/thesis_tools). Data preparation steps can also be found in this repository, it is advised to first follow the steps there.

## Installation
To install on the Lisa cluster: 

1. Load modules
   ```
   module load 2020
   module load Anaconda3/2020.02 
   ```

2. Clone the repo:
   ```
   git clone https://github.com/sanderisbestok/detectron2
   ```

3. Create the environment:
   ```
   conda create --name detectron python=3.8
   ```

4. Install packages
   ```
   conda install -c conda-forge opencv
   pip3 install torch torchvision torchaudio
   ```    

5. Load modules again
   ```
   module load 2020
   module load CUDA/11.0.2-GCC-9.3.0
   ```

6. Install detectron
   ```
   python -m pip install -e .
   ```

## Training
To train we need the pre-trained TridentNet model which can be downloaded [here](https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_101_C4_3x/148572198/model_final_164568.pkl). Place this model in the main folder.

The following job can be used to train the network if the network is installed in ~/networks/detectron2 with the environment yolov5.

```
#!/bin/bash
#SBATCH -t 06:00:00

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=gtx1080ti:4

module load 2020
module load CUDA/11.0.2-GCC-9.3.0
module load OpenCV/4.5.0-fosscuda-2020a-Python-3.8.2
module load Anaconda3/2020.02 

mkdir $TMPDIR/sander
cp -r $HOME/data $TMPDIR/sander/

source activate /home/hansen/anaconda3/envs/detectron/
cd ~/networks/detectron2/

python train_trident.py --num-gpus 4
```

## Validation & Testing
As of this moment, validation is build into the training stage. So during the training the results will be saved. 

### Testing
To do the actual testing on a test database instead of validation you can use the following command in the demo folder.

```
python test.py --config-file ../projects/TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml --input ~/data/extremenet/images/test/ --confidence-threshold 0.000000001 --opts MODEL.WEIGHTS ~/weights/experiment_1/trident_2499.pth MODEL.ROI_HEADS.NUM_CLASSES 1
```



## Extra
Detectron2 visualiser can be used with the following command:

Demo Commando:
python demo.py --config-file ../projects/TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml --input path_to_image --opts MODEL.WEIGHTS paht_to_model

The model is saved to the tridentnet_training_output folder inside the detectron2 repo.

