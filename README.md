# Sheeko Project

Machine learning implementation package to generate descriptive metadata for digitized historical images. The project is funded through the LYRASIS Catalyst grant.

## Contents
* [Project Overview](#Project-OverView)
    * [Introduction](#introduction)
    * [Architecture](#architecture)
* [Getting Started](#getting-started)
    * [Prerequisites](#Prerequisites)
    * [Installation](#Installation)
    * [Test Installation](#Test-Installation)
* [How to Use](#How-to-Use)
* [Walkthrough](#Walkthrough)
* [Prepare you Data](#Data-Preparation)
    * [Clean your data](#Clean-Data)
    * [Build your data](#Build-Data)
    * [Build TF Records](#Build-TF-Records)
* [Training a Model](#training-a-model)
    * [Initial Training For Caption Model](#initial-training)
    * [Fine Tune Caption Model With Pretrained Model](#fine-tune-the-im2txt-model)
* [Evaluate your Caption Model](#Evaluation-Caption-Model)
* [Inference](#Inference)
    * [Image Caption](#Image-Caption)
    * [Image Classification](#Image-Classification)
    * [Object Detection](#Object-Detection)
* [Pretrained Models](#Sheeko-Pretrained-Models-Resource) 



## Project Overview

This project provides deployable machine learning environment including pre-trained models and code packages for training, inference and evaluation for generating caption metadata. Inference codes for label generating (classification, object detect) are also included in this project.

### Introduction

Inference Example for given image

<img src="sheeko_demo_3.jpg" height="240" width="360">

Caption: 
  ```
  a bird sitting on top of a wooden bench . (p=0.001655)
  ```

Label: 
```
jay
indigo bunting, indigo finch, indigo bird, Passerina cyanea
magpie
pill bottle
water ouzel, dipper
Bird
```
### Architecture

This code package is the customization of Im2txt. Please see https://github.com/tensorflow/models/tree/master/research/im2txt for more details about im2txt


Architecture of Caption generating model in this project is ["Show and Tell model"](http://arxiv.org/abs/1609.06647) which is an encoder-decoder neural network.


<p align="center"><img src="show_and_tell.JPG" height="300" width="650"> </p>


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

There're two ways deploying this project on your platform. 
* 1. Deploy locally. Windows and Linux OS are supported.
* 2. Deploy in VirtualBox

Difference between these two ways:
* Deploy locally could maximize your GPU acceleration. This is useful when performing **training** and **evaluation**
* Deploy in VirtualBox could run the script without doing additional installation and for sure it's cross-platform.

You could deploy the project in both ways so you could get best performance in all functionalities.


#### To deploy locally, please install required packages

<ul>
<li><p>Ubuntu 16.04.6 LTS OS environment or Windows OS</p></li>
<li><p>NVIDIA GPU and NVIDIA Driver installed</p></li>
<li><p>Python 2.7 or Python 3</p></li>
<li><p>Tensorflow (For GPU version, please see https://www.tensorflow.org/install/gpu to install cuda related packages)</p></li>
<li><p><a href="https://www.nltk.org/">Natural Language Toolkit (NLTK)</a></p></li>
<li><p>Natural Language Toolkit (NLTK) punkt package</p></li>   
<li><p><a href="https://spacy.io/">spaCy</a></p></li>
<li><p><a href="https://pypi.org/project/python-resize-image/">python-resize-image</a></p></li>
<li><p>Pre-trained statistical NLP model: <a href="https://spacy.io/models/en">en_core_web_sm</a></p></li>
<li><p><a href="https://pypi.org/project/singledispatch/">singledispatch</a></p></li>   
<li><p><a href="https://pypi.org/project/contextlib2/">contextlib2</a></p></li>
<li><p><a href="https://pypi.org/project/Cython/">Cython</a></p></li>
</ul>

#### To deploy in Virtualbox

<ul>
   <li><p>Virtualbox Installed</p></li>
   <li><p>vt-x enabled</p></li>
   <li><p>Vagrant Installed</p></li>
</ul>

### Installation

#### To Install Locally

It's recommened to run the code in **Ubuntu 16.04.6 LTS OS or Windows OS environment**. 
Make sure you have **tensorflow-gpu and CUDA tookit** installed to enable GPU accleration. Installation for tensorflow-gpu is not covered in this document.

* Run the script to see python version. Make sure python is installed properly.

```
python -V
```

<ul>
   <li><p>For Windows OS</p>
   <p>Install the required packages listed above. </p>
   <p>To install punkt, run python script:</p>
   <p>
      
   ```  
   import nltk
   nltk.download("punkt")
   ```
   
   </p>
   </li>   
   <li><p>For Linux OS</p>
   <p>
      if you have the permission for sudo, run script under project root directory:</p>
     <p>  
        
```
bash installation.sh
```

 </p> 
 <p>
Otherwise install the required packages listed same as Windows OS.
 </p>      
</li>
</ul>
     
#### Use VirtualBox

This code package is supported using vagrant. **Virualbox** and **vt-x** option need to be enabled in BIOS mode. 

Please be aware that there is limitation for GPU allocation in VirtualBox environment. So It's not recommended to run either training or evaluation script in VirtualBox. 

Alternatively, if you plan to run training or evalutaion, please deploy the code package locally. 

* Download and install **[vagrant](https://www.vagrantup.com/downloads.html)** on your local environment. 

* Download code package and run the script under project root directory through command line
```
vagrant up
```

it will take few minutes if you running this script at the first time. It will create the VM and install all pakcages on the VM.

After the script is done, the vagrant box is up, and you may notice that VM "sheeko_project" is created. 
OS inside the VM is **Ubuntu 16.04.6 LTS OS** 

### Test Installation

If you deploy the code project in VM, you need to check installation inside the VM. Otherwise you can skip this step.
* 1.Start up and enter vagrant box 
```
vagrant up
vagrant ssh
```

* 2.Run following scripts to test packages installation
```
python -c 'import tensorflow as tf; print(tf.__version__)'
```
Version of the tensorflow will be printed as result if successfully installed
```
python -c 'import nltk; print(nltk.__version__)'
```
Version of the nltk will be printed as result if successfully installed

## How to Use

### Execute script
This code package is executed through Command Line. To run the script, you need to navigate to the directory of the script then run the script using "python script_name.py".

Here's an example:

```
cd /path/to/project/dir/data_preparation
python validate_data_run.py
```

### Configure File
To make the script work, you need to configure the scripts with the information needed.

Basically you need to configure the fields within the block in each script.

Here's an example:

```
-------------------configuration start here-----------------------------------------------------------------------
# Field name in annotation file containing metadata
FIELD_NM = "caption"
# Field name in the annotation file containing the image file name
FIELD_ID = "image_id"

# List of paths (relative or absolute) of directories containing the annotation files
CAPTION_DIR_LIST = ['clean/demo_1', 'clean/demo_2']

# Path to output directory
OUTPUT_PATH = "build"

# Segment method: seg_by_image, seg_by_dir
SEG_METHOD ='seg_by_image'

# Training set percent in int
TRAIN_PERCENT = 80
-------------------configuration end here-----------------------------------------------------------------------
```


If you plan to run the script under **Windows OS**, path format with "\\". 

e.g. CHECKPOINT_PATH = "path\\to\\dir\\pretrained_model\\graph.pb"


If **Linux OS**, path format with "/". 

e.g. CHECKPOINT_PATH = "path/to/dir/pretrained_model/graph.pb"

You could easily switch between Linux OS and Windows OS by replacing "\\" with "/" or replacing "/" with "\\" .


## Walkthrough

<h3>What's included in the package</h3>
<ul>
   <li>Data Preparation</li>
   <li>Training</li>
   <li>Evaluation</li>
   <li>Inference</li>
</ul>

In this section we're going to demo [how to perform the inference](#Inference-Walkthrough) and [how to get your own model](#Train-your-own-model-Walkthrough)

To make the script work, we need to configure the code file.

Please see [configuration](#Configure-File) for details.

### Inference Walkthrough

#### First we need to get the pretrained models, you could skip this if you have trained your own model already.

See our [pretrained models](#Sheeko-Pretrained-Models-Resource) website to download our pretrained models. 

In our walkthrough we use the [ptm-im2txt-incv3-mlib-cleaned-3m](https://sheeko.org/downloads/ptm-im2txt-incv3-mlib-cleaned-3m.zip) which is trained on MSCOCO dataset (http://cocodataset.org) with 3 million steps.


#### Next let's get the test images for inference. 

In our code package we provide a few demo images under <i>test_image</i> directory. Here we're using the demo images for inference.

#### Caption Inference

Let's start with turn on vagrant box

```
vagrant up

vagrant ssh
```

Now let's do the the caption inference.

the project directory is also the shared folder with the Virutalbox and that means you could edit the file either inside the VM or locally. In VM, the path the project is under <i>/vagrant/</i>.

For example: in vagrant box,
Open <i>/vagrant/inference/caption/caption_inference_test.py</i> and configure the following fields:

* 1. CHECKPOINT_PATH. Path to the directory containing the checkpoint file (pre-trained model)
* 2. VOCAB_FILE.  Path to the Vocabulary Dictionary file
* 3. IMAGE_FILE. Path to the JPEG image file to generate caption

You could also locate the file under the <i> project_directory/inference/caption/caption_inference_test.py </i>

Here's an example of configuration under linux OS (vagrant).
```
'''Path to the directory containing the checkpoint file (pre-trained model)'''
CHECKPOINT_PATH="../../models/caption/mscoco2017_3M/train"

# Vocabulary Dictionary file generated by the preprocessing script.
VOCAB_FILE = "../../models/caption/mscoco2017_3M/word_counts_mscoco.txt"

# Path to the JPEG image file to generate caption
IMAGE_FILE = "../../test_image/sheeko_demo_4.jpg"
```

Then we need to navigate to caption_inference_test located directory and run the script

```
cd /vagrant/inference/caption

python caption_inference_test.py
```
So we get the following results printed
```
  0) a cow standing in a field of grass . (p=0.000430)
  1) a cow is standing in a field of grass . (p=0.000424)
  2) a cow standing in a field of grass (p=0.000283)
```

caption_inference_run.py has the similar fields to configure. There's slight diference that it can loop over all images under the each directory in the list and generate the captions. Also instead of printing out the results, it creates a JSON file containing of the results generated.

Here's an example of configuration under linux OS (vagrant).

```
# List of paths to directories containing JPEG image files to caption. 
# The script will grab all JPEG in specified directories. 
# No need to mention files individually. 
# This will only grab images place directly in the directories. It will not go into child dirs.
IMAGE_DIR_LIST = ["../../test_image"]

# Path to output json file
OUTPUT_PATH = "output/captions.json"
```

After running the script, the results will be stored in JSON file located at <b>OUTPUT_PATH</b>.

Feel free to do inference with your own images. However, this model is not perfect for doing inference for all images. 

e.g. the  sheeko_demo_1.jpg 
the result generated says it's a bird standing on the top of the wooden branch.
<img src="sheeko_demo_1.jpg" height="240" width="360">


#### Classification Inference

https://github.com/googlecodelabs/nest-tensorflow/tree/master/tmp/imagenet

Now let's do the the classification labels inference.

Open <i>/vagrant/inference/classification/classify_image_test.py</i> and configure the following fields:

* 1. CHECKPOINT_PATH. Path to the directory containing the checkpoint file (pre-trained model)
* 2. VOCAB_DIR. Path to the directory containing Vocabulary Dictionary file
* 3. IMAGE_FILE. Path to the JPEG image file to generate caption

You could also locate the file under the <i> project_directory/inference/classification/classify_image_test.py </i>

Here's an example of configuration under linux OS (vagrant).
```
# Path to pretrained model graph.pb file
CHECKPOINT_PATH = "../../models/labels/classification/classify_image_graph_def.pb"

# Path to vocabulary directory that containing pbtxt and txt dictionary file.
VOCAB_DIR = "../../models/labels/classification"

# Path to the JPEG image file to generate label
IMAGE_FILE = "../../test_image/sheeko_demo_4.jpg"
```

Then we need to navigate to classify_image_test located directory and run the script

```
cd /vagrant/inference/classification

python classify_image_test.py
```
So we get the following results printed
```
ox (score = 0.89056)
oxcart (score = 0.03067)
water buffalo, water ox, Asiatic buffalo, Bubalus bubalis (score = 0.00574)
hog, pig, grunter, squealer, Sus scrofa (score = 0.00301)
lakeside, lakeshore (score = 0.00288)
```
similar to caption_inference_run.py, classify_image_run.py also stored generated results in JSON File.

Here's an example of configuration under linux OS (vagrant).

```
# Path to pretrained model graph.pb file
CHECKPOINT_PATH = "../../models/labels/classification/classify_image_graph_def.pb"

# Path to vocabulary directory that containing pbtxt and txt dictionary file.
VOCAB_DIR = "../../models/labels/classification"

# List of paths to JPEG image files to labels
# The script will grab all JPEG in specified directories. 
# No need to mention files individually. 
# This will only grab images place directly in the directories. It will not go into child dirs.
IMAGE_DIR_LIST = ["../../test_image"]

# Path to output json file
OUTPUT_PATH = "output/classifications.json"
```

After running the script, the results will be stored in JSON file located at <b>OUTPUT_PATH</b>.


#### Object Detection Inference


Now let's do the the object detect labels inference.

Open <i>/vagrant/inference/object_detect/object_detect_test.py</i> and configure the following fields:

* 1. CHECKPOINT_PATH. Path to the directory containing the checkpoint file (pre-trained model)
* 2. VOCAB_FILE. Path to Vocabulary Dictionary file
* 3. IMAGE_FILE. Path to the JPEG image file to generate caption

You could also locate the file under the <i> project_directory/inference/object_detect/object_detect_test.py </i>

Here's an example of configuration under linux OS (vagrant).
```
# Path to pretrained model graph.pb file
CHECKPOINT_PATH = "../../models/labels/object_detect/models/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb"


# Path to label mapping pbtxt file
VOCAB_FILE = "../../models/labels/object_detect/vocab/oid_bbox_trainable_label_map.pbtxt"

# Path to the JPEG image file to generate label
IMAGE_FILE = "../../test_image/sheeko_demo_4.jpg"
```

Then we need to navigate to object_detect_test located directory and run the script

```
cd /vagrant/inference/object_detect

python object_detect_test.py
```
Since object detect is GPU usage consuming, it's slower to run it under vagrant box. If you plan to do inference for multiple images,
it's recommended to run the script locally.

Here's the result:
```
[{'score': '0.72273505', 'label_text': u'Cattle'}, {'score': '0.3564907', 'label_text': u'Animal'}, {'score': '0.34420493', 'label_text': u'Furniture'}]

```
similar to caption_inference_run.py, object_detect_run.py also stored generated results in JSON File.

Here's an example of configuration under linux OS (vagrant).

```
# Path to pretrained model graph.pb file
CHECKPOINT_PATH = "path/to/dir/pretrained_model/graph.pb"

# Path to label mapping pbtxt file
VOCAB_FILE = "path/to/dir/pretrained_model/label_map.pbtxt"

# List of paths to JPEG image files to caption. It's no longer needed since image_dir is used to grab all images inside to generate the captions, for legacy testing, please go to research/im2tx test_model.py
IMAGE_DIR_LIST = ["path/to/dir/"]

# Path to output json file
OUTPUT_PATH = "path/to/output/object_detect.json"
```

After running the script, the results will be stored in JSON file located at <b>OUTPUT_PATH</b>.


### Train your own model Walkthrough

#### The first thing for your training is always about the data. 

Please see [Data Prepartion Section](#Data-Preparation) for details.
Here to make the demo faster, we use the provided data under <i>data_prepartion/data</i>. Under data directory both annotation files and JPEG images are provided.

#### Clean and Structure your data

Most time the data you get is always not clean, so cleaning actually takes most of the time for your training. Here we provide the code to do the trick. It cleans up your data by removing proper-noun (See [Clean Data](#Clean-Data) for how it works) and also structures your data into training and testing sets. (See [Build Data](#Build-Data) for how it works)

You could skip data cleaning if you already have the clean data.

Open <i>/vagrant/data_preparation/clean_data_run.py</i> and configure the following fields:

* 1. FIELD_NM. Field name in annotation file containing metadata
* 2. DATA_DIR_LIST. List of directories containing annotation files that will process data cleaning
* 3. OUTPUT_PATH. Path to output directory 
* 4. DATA_DIR. This is alternative way for providing list of directories, you could loop over the given root directory

You could also locate the file under the <i> project_directory/data_preparation/clean_data_run.py </i>

Here's an example of configuration under linux OS (vagrant). Here we loop over data directory, so all directories under "data" will be processed. 

```
# Field name in annotation file containing metadata
FIELD_NM = "description_t"

# Example of looping all dirs under given path
DATA_DIR = 'data'

for dir in os.listdir(os.path.abspath(DATA_DIR)):
    if os.path.isdir(os.path.abspath(os.path.join(os.path.abspath(DATA_DIR),dir))):
        DATA_DIR_LIST.append(DATA_DIR + '/' + dir)

# Path to output directory
OUTPUT_PATH = "clean"
```

Then we need to navigate to clean_data_run located directory and run the script

```
cd /vagrant/data_preparation/

python clean_data_run.py
```

After running the script, the clean annotation files will be stored under <b>"clean"</b>.


Once we've got the clean data needed. We could now structure our data into the format for training.

Open build_data_run.py and configure the fields:
* 1. FIELD_NM. Field name in annotation file containing metadata
* 2. FIELD_ID. Field name in annotation file containing image id matching image file name.
* 3. IMAGE_DIR_LIST. List of directories containing image files.
* 4. CAPTION_DIR_LIST. List of directories containing annotations files.
* 5. OUTPUT_PATH. Path to output directory 
* 6. DATA_DIR. This is alternative way for providing list of directories, you could loop over the given root directory
* 7. SEG_METHOD. Segment method: seg_by_image, seg_by_dir.  
* 8. TRAIN_PERCENT. Integer of percent to put into training data set. The rest of data will go the test set.

Here we put the cleaned annotations file along with the image files in the build_data_run.py.

```
# Field name in annotation file containing metadata
FIELD_NM = "caption"
# Field name in the annotation file containing the image file name
FIELD_ID = "image_id"

# List of paths (relative or absolute) of directories containing the annotation files
CAPTION_DIR_LIST = ['clean/demo_1', 'clean/demo_2']
# List of paths (relative or absolute) of directories containing the images
IMAGE_DIR_LIST = ['data/demo_1', 'data/demo_2']

# Path to output directory
OUTPUT_PATH = "build"

# Segment method: seg_by_image, seg_by_dir
SEG_METHOD ='seg_by_image'

# Training set percent in int
TRAIN_PERCENT = 80
```

Run the script:

```
python build_data_run.py
```

The structured data is stored under <b>"build"</b>


#### Convert Data into TF Records

To make data runnable by training script, we need to convert the images and captions into TF records which are serial image-caption pairs.

Open <i>/vagrant/data_preparation/build_TF_run.py</i> and configure the following fields:

* 1. TRAIN_SET_IMAGE. Path to directory containing training set images
* 2. TEST_SET_IMAGE.Path to directory containing testing set images
* 3. TRAIN_SET_CAPTION. Path to training set annotation file
* 3. TEST_SET_CAPTION. Path to testing set annotation file
* 4. OUTPUT_PATH. Path to output directory 

You could also locate the file under the <i> project_directory/data_preparation/build_TF_run.py </i>


```
# Path to directory containing training set images
TRAIN_SET_IMAGE = "build/train/images"
# Path to directory containing testing set images
TEST_SET_IMAGE = "build/test/images"
# Path to training set annotation file
TRAIN_SET_CAPTION = "build/train/annotations/annotation.json"
# Path to testing set annotation file
TEST_SET_CAPTION = "build/test/annotations/annotation.json"

# Path to output directory
OUTPUT_PATH = "TF"
```

Then we need to navigate to build_TF_run located directory and run the script

```
cd /vagrant/data_preparation/

python build_TF_run.py
```

After running the script, the TF Records will be stored under <b>"TF"</b>.



Run training 

Get test images

Run Inference

Run Evaluation

To demo the Inference for pretrained models, Please go to [Inference](#Inference)

To train your own model for generating caption, Please start with [Data Preparation](#Data-Preparation) 

#### Get Started
If you installed the packaged locally,
* Skip the first step and following the rest of steps
If you installed the packaged using vagrant box,
* 1. login into vagrant box
```
vagrant ssh
```
* 2. migrate annotation files that need to clean into project directory.

e.g. data_preparation\data contains demo data to perform NLP cleaning

* 3. navigate to data_preparation directory and open clean_data_run.py file and configure the following setting
```
# Field name in annotation file containing metadata
FIELD_NM = "description_t" 
e.g. "{
    "date_t": 1943,
    "description_t": "Brooch made by internees at the Topaz Internment Camp in Delta, Utah. Residents would gather sea shells found at the camp, bleach them, and then paint and arrange them into brooches and other designs.",
    ...
    "

# List of paths (relative or absolute)of directories containing the meta data that needs cleaning
# This will perform NLP cleaning and remove proper-nouns
DATA_DIR_LIST = ['data/demo_1']

# or using Loop function to perform NLP on all sub directories under the given directory
DATA_DIR = 'data'


# Path to output directory
OUTPUT_PATH = "clean"
```

## Data Preparation
Data from any source for caption generating model need following the required pattern.

Both images files in JPEG format and descriptive annotation files in json format are required.
* JPEG image
* JSON annotation file containing metadata of caption associated with JPEG image

#### JSON Annotation File Example
```

[
images:[{
"file_name": "/vagrant/data/dc_bpc/125853.jpg", "id": 57870
}, ...
]
annotations:[{  id: 1,  image_id: 57870,  file_name: "COCO_train2014_000000000009.jpg"  caption: "a cat and dog on wooden floor next to a stairwell."}, ...
]

```
[The sheeko pretrained models](#Sheeko-Pretrained-Models-Resource) are trained using the data source below:

* 120 K images with metadata in MSCOCO dataset (http://cocodataset.org)
* 470 K images with metadata in J. Willard Marriott Digital Library Collections ( https://collections.lib.utah.edu/)

### Clean Data
This step is not required if your data set is clean and doesn't contain any noise (e.g. proper-noun). In this section we're removing proper-noun type noises using spaCy based NLP process in data set.

#### Toolkit

API: spaCy

Model: [en_core_web_sm](https://spacy.io/models/en)

#### What does it do?

* Remove proper-noun noise

* Simplify annotation file by leaving image_id and metadata field only

* Save cleaned annotation file in the destination directory following the same directory structure

* Create data report in JSON format under destination directory

    * **valid metadata**: total number of captions that have valid JPEG images found and got non empty result after NLP process
    * **dir name**: /path/to/data/dir, 
    * **none pronoun**: total number of captions that have valid JPEG images found and are proper-noun noise free 
    * **pronoun**: total number of captions that have valid JPEG images found and have proper-noun noise
    * **data count**: total number of captions that have valid JPEG images found

#### Neuro-linguistic programming (NLP) Steps
##### step 1: PRONOUN cleaning
* 1.1 replace named person with "person"

example: Mary Jane with daughters --> a person with daughters
* 1.2 remove DATE, TIME, ORG, GPE, LOC, PRODUCT, EVENT, WORK ART, LAW, LANGUAGE, FAC, NORP, PERCENT, MONEY, QUANTITY, ORIGINAL     
and get their parent node, if there's no parent node, then abort this sentence)

example: A women group is photographed at the Ashley Power Plant up Taylor Mountain road --> A women group is photographed
##### step 2: Parent nodes cleaning 
* 2.1 if the parent node is 'DET', 'NOUN', 'NUM', 'PUNCT', 'CCONJ', 'VERB', remove the subtrees nodes of the lefts if it's not in the list 
* 2.2 if the parent node is PREP or other types, remove the nodes and their children nodes

example: A woman tends to a fire at a Ute Indian camp near Brush Creek. --> A woman tends to a fire at a camp .

##### step 3: NOUN entities cleaning
* 3.1 replace the chunk of non-named enties chunk with simple entity along with CC, CD, DT and IN
* 3.2 replace CD type number with string value, e.g. 3, three

example: Ute couple with child stand in front of old cars --> couple with child stand in front of cars

##### step 4: Reform
* 4.1 put person entity and chunk replacement in position
* 4.2 replace person entities with understandable words, e.g. ,if multiple people then use num people instead, e.g. "a person" "two people", else say "a group of people"
* 4.3 replace nums of person entities (>=3) replace with "a group of " + noun
* 4.4 remove the tokens with index in indexes list of the sentence and convert the array to sentence 


### Build Data

#### Description
Formatted data in structure is required for the package. To create format data we need to go through build data section.

#### What does it do?
* Filter data set by getting data that have both images and associating annotation only
* Segment images and annotation files into training and testing data sets
* Resize image to trainable format into training and testing data set
* Build annotation file for training and testing data set

##### Segment Mode
We provide two segment methods:

1. segment all data through out all directories as one into training and testing data set.

2. segment the list of directories into training and testing sets.

* seg_by_image
segment data set in each directory with given percentage of data into training set, the rest into test


* seg_by_dir
segment directories with given percentage into training set, the rest of directories into test

#### Steps
* Make clear the purpose: Training, Inference or Evaluation
* Prepare your data according to the purpose
* Make sure you have enough space in the output directory to store the output result ( 6 X of original data set in file size)

#### For particular purpose
##### For Training
* Prepare images file in JPG format (single file size no more than 15 mb) and associating annotation files in JSON format. 
Structure your data: image file name has to be unique id and annotation file has to has the same image_id of image file name (e.g. 15376.jpg,  annotation: "image_id": 15376 )
* Structure your data: annotation files have to use the consistent field for getting metadata (e.g. "description_t")

##### For cleaning up proper-noun and data noise in your annotation file by using clean_data_run.py and configure field name of metadata field, list of paths to annotation files and output directory, for more details, please see Data Cleaning page.
* Configure arguments in build_data_run.py (field names in annotation file, list of paths to annotation files and image files, output directory and data segmentation args, e.g. method and training set percentage )
Run build_data_run.py
* Formatted data will be available in the directory you specified as OUTPUT_PATH in build_data_run.py
* For im2txt captioning model training. Run build_TF_run.py to generating TF Records. Each image object contains "file_name" and "id". Each annotation object contains "id", "image_id" and "caption". Each image object may refers to multiple caption objects
* TF Records will be located under OUTPUT_PATH which is runnable data for the training

##### For Inference
* Prepare images file for inference in JPG format
* Prepare checkpoint file of model and corresponding vocabulary file 
* Configure arguments in build_data_run.py (field names in annotation file, list of paths to annotation files and image files and output directory)
* Run build_data_run.py
* Inference results will be stored in OUTPUT_PATH in caption_infer.py 


### Build TF Records    
To make data runnable by training script, data need to be converted into TF Record format. Each TF record represents serial pairs of encoded image-caption data. Please see [Convert Data into TF Records](#Convert-Data-into-TF-Records) in Walkthrough section to see details.

   
## Training a Model

### Initial Training
    
### Fine Tune the im2txt Model 
## Evaluation Caption Model

### Evaluation Caption Model
##### Metric
Caption: perplexity per word

## Inference

### Image Caption
 
### Image Classification
 
### Object Detection
 
## Sheeko Pretrained Models Resource
This project provides [Sheeko Pretrained Models Resource](https://sheeko.org/pre-trained-models/) for generating captions.
Models with description are available for downloading. It's highly recommended to try the downloaded model in [Inference](#Inference) to test the performance.

## Windows and Linux switch (path / for linux or \\ for windows)
## Data built in linux environment could not be used in Windows, you have to run data build in windows environment to build the format for windows
## Installation script implementation for windows and linux (linux run installation.sh, windows run installaation_win.sh)
## GPU Use (run locally, specify which gpu)
## Metadata field (example of specifing metadata field and image id for cleaning, data build)
## Model Migration (clear absolute path to make relative)
## Provide resources downloading inception v3 model, classified label model and object detect model

