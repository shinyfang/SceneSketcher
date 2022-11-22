# SceneSketcher-v2
This repository contains the datasets and the code for training the model. Please refer to our TIP 2022 paper for more information: ["SceneSketcher-v2: Fine-Grained Scene-Level Sketch-Based Image Retrieval using Adaptive GCNs. "](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779565)

## Dataset
We modified existing sketch databases SketchyCOCO and SketchyScene for evaluations. The datasets consist of three parts:
* SketchyCOCO-SL (train 1015 + test 210)
* SketchyCOCO-SL Extended (train 5210 + test 210)
* SketchyScene (train 2472 + test 252)


## Evaluation
### Download a pre-trained model 
The pretrained model is saved at [Google Drive Hosting](https://drive.google.com/file/d/1pClJS6hBe70xt1vT1_5oodeLZwCF84pZ/view)

For evaluation under `SketchyCOCO-SL` dataset, run:
```
python evaluate_attention.py
```

## Training
### Preparations
You can train your own model by processing the training data into the format of the data in the `test` folder.
`GraphFeatures` is composed of the category and position of each instance in a scene. We adopt Inception-V3 trained on ImageNet to extract a 2048-d feature for each instance.

After the preparations, run:

```
python train_attention.py
```

## Citation
```bibtex
@ARTICLE{9779565,
  author={Liu, Fang and Deng, Xiaoming and Zou, Changqing and Lai, Yu-Kun and Chen, Keqi and Zuo, Ran and Ma, Cuixia and Liu, Yong-Jin and Wang, Hongan},
  journal={IEEE Transactions on Image Processing}, 
  title={SceneSketcher-v2: Fine-Grained Scene-Level Sketch-Based Image Retrieval Using Adaptive GCNs}, 
  year={2022},
  volume={31},
  number={},
  pages={3737-3751},
  doi={10.1109/TIP.2022.3175403}}
```
