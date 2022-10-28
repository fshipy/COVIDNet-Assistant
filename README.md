# COVIDNet-Assistant

<p align="center">
	<img src="image/covidnetlogo.png" alt="photo not available" width="30%" height="30%">
	<br>
	<em></em>
</p>


This code is for the paper **COVID-Net Assistant: A Deep Learning-Driven Virtual Assistant for Early COVID-19 Recommendation** 

**Link to the [paper](https://arxiv.org)[add link]**

<p align="center">
	<img src="image/covidnet_assitant_workflow.png" alt="photo not available" width="70%" height="70%">
	<br>
	<em>Figure 1: Overview  of COVID-Net Assistant workflow.</em>
</p>

**In this repository, we present covid cough models to demonstrate the effectiveness and speed of covid detection neural networks trained on cough datasets.**

## Abstract

As the COVID-19 pandemic has addressed the interests of the public in recent years, researchers are interested in finding effective and inexpensive screening methods to leverage the use of medical resources. The most common symptoms of COVID-19 are fever, fatigue and dry coughs. Coughs have been used as a informal screening tool for a variety of diseases and we believe it can serve as a screening tool for COVID-19 detection as well. In this study, we introduce the design of COVID-Net Assistant, an efficient virtual assistant to provide status prediction and early recommendations for  COVID-19 by processing users' cough sounds through deep convolutional neural networks. We present multiple lightweight convolutional neural network architectures built with standard convolutions, residual blocks, and depth-wise separable convolutions. We trained and evaluated the COVID-Net Assistant neural networks on the Covid19-Cough dataset, which includes 381 pieces of cough audio with positive labels verified by PCR test. Our experimental results show COVID-Net Assistant neural networks have robust performance on the dataset split, excluding unverified positive samples. On average, our models can achieve over AUC score of 0.93, with the best score over 0.95 while being fast and efficient in inference. © 2022 .


## Table of Contents
1. [Requirements to install on your system](#requirements)
2. [How to download and prepare the Covid19-Cough dataset](docs/dataset.md)
3. [Steps for training, evaluation and inference](docs/train_eval_inference.md)
4. [Results and links to pretrained models](docs/models.md)


#### Requirements
The main requirements are listed below:

* Tensorflow 1.15
* librosa
* audiomentations
* keras
* Python 3.7
* Numpy
* Scikit-Learn
* Matplotlib
* ffmpeg
* ipywebrtc

Please use this script to insall all requirements
```
$ sudo apt install ffmpeg
$ pip3 install -r requirements.txt
$ jupyter nbextension enable --py widgetsnbextension
```

#### Contact

If there are any technical questions after the README, FAQ, and past/current issues have been read, please post an issue or contact:

* p23shi@uwaterloo.ca
* y3222wan@uwaterloo.ca

#### Citation (TODO add citation)

```
@article{

}

```
