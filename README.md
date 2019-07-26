# Sheeko Project

Generate descriptive metadata for digitized historical images. The project is funded through the LYRASIS Catalyst grant.

## Contents
* [Project Overview](#Project-OverView)
    * [Introduction](#introduction)
    * [Architecture](#architecture)
* [Getting Started](#getting-started)
    * [Prerequisites](#Prerequisites)
    * [Installing](#Installing)
    * [Test Installation](#Test-Installation)
* [Pretrained Models](#Pretrained-Models)
* [Prepare you Data](#Data-Preparasion)
* [Training a Model](#training-a-model)
    * [Initial Training](#initial-training)
    * [Fine Tune the Inception v3 Model](#fine-tune-the-inception-v3-model)
* [Inference](#Inference)
* [Evaluate your model](#Evaluation)

## Project Overview
### Inference Example

Caption: 
  ```
  0) a large building with a clock tower on top . (p=0.000486)
  1) a large building with a clock tower in the middle . (p=0.000378)
  2) a large building with a clock tower in the middle of it . (p=0.000202)
  ```

Label: Tree, Street light, Pole
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Recommened Ubuntu 16.04.6 LTS OS
Other OS are supported with virtualbox installed and vt-x option enabled in BIOS mode
```

### Installing

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

