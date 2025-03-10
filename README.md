# ML_KWS
ML-Tool for Keyword Spotting, which includes data collection with EVB, training, and conversion.

# 1. First step
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Please download and install the Nu-Link driver from the following link [KEIL Nu-Link debugger driver installer](https://github.com/OpenNuvoton/Nuvoton_Tools)
- (Improtant (11/2 updated)) If you would like to use `ML_audio_record`, please make sure that `PyYAML` is at version `6.0` due to `pyocd` dependency. However, if you have already installed [ML_Object_Detection](https://github.com/OpenNuvoton/ML_Object_Detection), `PyYAML` will be re-installed to version `5.4.1`, so you will need to run `pip install PyYAML==6.0` again. During this process, you may encounter an error stating that `tf-models-official` conflicts with the PyYAML version. Please ignore it, as it will not affect the usage of the [ML_Object_Detection](https://github.com/OpenNuvoton/ML_Object_Detection) or [ML_KWS](https://github.com/OpenNuvoton/ML_KWS) tool.
---
# 2. Work Flow
 <img src="https://user-images.githubusercontent.com/105192502/202999518-7d4a6384-6cef-4901-b948-b1117baa7bdd.png" width="50%">

## A. Collect your own KWS audio raw data in `ML_audio_record` folder
- Open `record_mcu.ipynb`.
- This notebook will assist you in loading (flashing) a record function *bin file to your m460 board and recording your voice.
- Please leave at least a 1-second gap between each keyword and continue collecting raw data until you have accumulated enough for training purposes.
- The raw data will be saved in the `raw` folder. You can move all the files to the same label folder for later preprocessing.
- (Note) You can also use Google's training dataset (Can be downloaded at training step) during the training step.

## B. Processing the raw data in `ML_audio_aq` folder
- Open `sound_crop.ipynb`.
- You can copy the previous label folder with raw data to `ML_audio_aq` for slicing each keyword individually.
- The sliced data will be saved in the `dataset\<YOUR_LABEL>` folder.

## C. Traing/Testing/Converting to Tflite in `ML_kws_tflu` folder
- Open `train.ipynb`, `test.ipynb`, `convert.ipynb`.
- The instructions on how to use these notebooks are described in the Jupyter Notebooks themselves.
- (Note) It is recommended to download the Google's training data initially and then move your own training data folders into the same Google train data folder.

## D. Vela Compiler on M55M1
- Move the int8 quantized model (In `ML_kws_tflu/work/YOUR_PROJECT` after training&converting) to `ML_kws_tflu\vela\generated`.
- update `MODEL_SRC_FILE=YOUR_INT8_TFLITE_NAME` and `MODEL_OPTIMISE_FILE=YOUR_VELA_TFLITE_NAME` in `variables.bat`.
- Execute the `gen_model_cpp.bat`. The vela tflite and c++ model file are in `ML_kws_tflu\vela\generated`.

# 3. Inference code
- The ML_SampleCode repositories are private. Please contact Nuvoton to request access to these sample codes. [Link](https://www.nuvoton.com/ai/contact-us/)
  - [ML_M460_SampleCode (private repo)](https://github.com/OpenNuvoton/ML_M460_SampleCode)
  - `tflu_kws_arm`/`tflu_kws_arm_mc`: offline evaluation inference with DNN/DS-CNN
  - `tflu_kws_arm_rt`/`tflu_kws_arm_rt_mc`: real time inference code with DNN/DS-CNN
  - `tflu_kws_record`: The example code for recording keywords, designed to pair with `record_mcu.ipynb`
- [M55M1BSP](https://github.com/OpenNuvoton/M55M1BSP/tree/master/SampleCode/MachineLearning)
    - KeywordSpotting/ 
 
  
 
