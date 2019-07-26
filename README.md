# Sheeko Project

Generate descriptive metadata for digitized historical images. The project is funded through the LYRASIS Catalyst grant.

## Contents
* [Project Overview](#Project-OverView)
    * [Introduction](#introduction)
    * [Architecture](#architecture)
* [Getting Started](#getting-started)
    * [Prerequisites](#Prerequisites)
    * [Installation](#Installation)
    * [Test Installation](#Test-Installation)
    * [What's Included in the Package](#What's-Included-in-the-Package)
* [Pretrained Models](#Pretrained-Models)
    * [ptm-im2txt-incv3-mscoco-1m](#ptm-im2txt-incv3-mscoco-1m)
    * [ptm-im2txt-incv3-mscoco-3m](#ptm-im2txt-incv3-mscoco-3m)
    * [ptm-im2txt-incv3-mlib-cleaned-1m](#ptm-im2txt-incv3-mlib-cleaned-1m)
    * [ptm-im2txt-incv3-mlib-cleaned-3m](#ptm-im2txt-incv3-mlib-cleaned-3m)
    * [ptm-im2txt-incv3-mlib-uncleaned-1m](#ptm-im2txt-incv3-mlib-uncleaned-1m)
    * [ptm-im2txt-mscoco1m-mlib-cleaned-2m](#ptm-im2txt-mscoco1m-mlib-cleaned-2m)
    * [ptm-im2txt-mscoco3m-mlib-cleaned-1m](#ptm-im2txt-mscoco3m-mlib-cleaned-1m)    
* [Prepare you Data](#Data-Preparasion)
    * [Clean your data](#Clean-Data)
    * [Build your data](#Build-Data)
    * [Build TF Records](#Build-TF-Records)
* [Training a Model](#training-a-model)
    * [Initial Training For Caption Model](#initial-training)
    * [Fine Tune Caption Model With Pretrained Model](#fine-tune-the-im2txt-model)
* [Inference](#Inference)
    * [Image Caption](#Image-Caption)
    * [Image Classification](#Image-Classification)
    * [Object Detection](#Object-Detection)
* [Evaluate your Caption Model](#Evaluation-Caption-Model)

## Project Overview
### Inference Example
![](demo4.jpg)
Caption: 
  ```
  0) a street sign on a pole on a street (p=0.000016)
  1) a street sign on a pole on a city street (p=0.000013)
  2) a street scene with focus on a train track . (p=0.000010)
  ```

Label: 
```
worm fence, snake fence, snake-rail fence, Virginia fence (score = 0.11539)
patio, terrace (score = 0.06765)
planetarium (score = 0.06152)
flagpole, flagstaff (score = 0.04749)
picket fence, paling (score = 0.04370)

[{'score': '0.8693373', 'label_text': u'Tree'}]
```
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
To enable GPU accleration for training and evaluation, it's recommened to run the code in Ubuntu 16.04.6 LTS OS environment

Other OS are supported with virtualbox installed and vt-x option enabled in BIOS mode. Please be aware that there is limitation for GPU allocation in VirtualBox environment. So It's not recommended to run either training or evaluation script in VirtualBox. 

Alternatively, please migrate the installation.sh, train, evaluation and library directories to local machine for the purpose. 
Run "bash installation.sh" under project root directory to install prerequisites packages before running training or evaluation scripts
```

### Installation

A step by step series of examples that tell you how to get a development env running

Say what the step will be

For Linux OS system
Run script under project root directory
```
bash installation.sh
```

For Windows or other OS
Run
```
vagrant up
```
## Test Installation
Run following scripts to test packages installation
```
python -c 'import tensorflow as tf; print(tf.__version__)'
```
```
python -c 'import nltk; print(nltk.__version__)'
```

## Running the tests

Model



### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc




installation
vagrant 
linux

test demo
5 images

