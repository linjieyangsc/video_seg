# OSMN: One-Shot Modulation Network for Semi-supervised Video Segmentation


### Installation:
1. Clone the repository
   ```Shell
   git clone https://github.sc-corp.net/linjie-yang/video_segmentation.git
   ```
2. Install if necessary the required dependencies:
   
   - Python 2.7 
   - Tensorflow r1.0 or higher (`pip install tensorflow-gpu`) along with standard [dependencies](https://www.tensorflow.org/install/install_linux)
   - Other python dependencies: PIL (Pillow version), numpy, scipy, matplotlib
   


### Pre-training the network on MS-COCO
1. Download MS-COCO 2017 dataset from [here](http://cocodataset.org/#download).
2. Download the VGG 16 model trained on Imagenet from the TF model zoo from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
3. Place the vgg_16.ckpt file inside `models/`.
4. Run `python osmn_vs_coco_pretrain.py --data_path DATA_PATH --result_path RESULT_PATH --model_save_path MODEL_SAVE_PATH --gpu_id GPU_ID` to train the model. Other arguments can be seen by running `python osmn_vs_coco_pretrain.py -h`.

#Fine-tuning the network on DAVIS
1. Download DAVIS 2017 dataset from [here](http://davischallenge.org/code.html).
2. Preprocess the dataset by running `python preprocessing/parseData.py DATA_DIR`.
3. Run `python osmn_vs_train_eval.py --data_path DATA_PATH --src_model_path SRC_MDOEL_PATH --result_path RESULT_PATH --model_save_path MODEL_SAVE_PATH --gpu_id GPU_ID`. Other arguments can be seen by running `python osmn_vs_train_eval.py -h`.

