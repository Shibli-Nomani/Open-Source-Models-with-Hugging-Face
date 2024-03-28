# Open Source Models with Hugging Face
**🤗 Hugging Face Overview:**
Hugging Face is a leading platform for natural language processing (NLP), offering a vast repository of pre-trained models, datasets, and tools, empowering developers and researchers to build innovative NLP applications with ease.

![alt text](<Hugging Face.png>)

### Github 👇
- **Text**
- **👉 chatbot github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/chatbot.ipynb
- **👉 text translation and text summarization github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/text%20translation%20and%20text%20summarization.ipynb
- **👉 sentence embedding github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/Sentence%20Embeddings.ipynb

- **Audio**
- **👉 audio classification real-world dataset github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/audio%20classification%20real-world%20dataset.ipynb
- **👉 Automatic Speech Recognitions and Gradio apps github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/Automatic%20Speech%20Recognitions%20and%20Gradio%20apps.ipynb
- **👉 Text to Speech with VITS-Conditional Variational Autoencoder github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/Text%20to%20Speech%20with%20VITS-Conditional%20Variational%20Autoencoder.ipynb

- **Image**
- **👉 Object Detection And Generate Audio Based On Detection github code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/Object%20Detection%20with%20detr-resnet-50%20and%20gradio.ipynb

#### Kaggle 👇
- **Text**
- **👉 chatbot kaggle code:** https://www.kaggle.com/code/shiblinomani/chatbot-with-hugging-face-model
- **👉 text translation and text summarization kaggle code:** https://www.kaggle.com/code/shiblinomani/text-translation-summarization-with-hugging-face/notebook
- **👉 sentence embedding kaggle code:** https://www.kaggle.com/code/shiblinomani/sentence-embedding-with-hugging-face-model/notebook

- **Audio**
- **👉 audio classification real-world dataset kaggle code:** https://www.kaggle.com/code/shiblinomani/audio-classification-real-world-dataset/notebook
- **👉 Automatic Speech Recognitions and Gradio apps kaggle code:** https://www.kaggle.com/code/shiblinomani/automatic-speech-recognitions-and-gradio-apps/notebook
- **👉 Text to Speech with VITS-Conditional Variational Autoencoder kaggle code:** https://www.kaggle.com/code/shiblinomani/text-to-speech-with-vits-auto-encoder/notebook
  


### 😸 Jupyter Notebook Shortcuts
```
https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330
```

### 🐍 Python Environment Setup
- Install vscode

- Install Python pyenv

- Install python 3.10.8 using pyenv (Pyevn cheat sheet added below)

- video link to install pyenv and python
```sh
    https://www.youtube.com/watch?v=HTx18uyyHw8
```
```sh
    https://k0nze.dev/posts/install-pyenv-venv-vscode/
```
ref link: https://github.com/Shibli-Nomani/MLOps-Project-AirTicketPricePrediction

##### Activate Python Env in VSCODE

```sh
   pyenv shell 3.10.8
```
##### Create Virtual Env

- #create directory if requires
```sh
mkdir nameoftheproject
```
- #install virtualenv
```sh
pip install virtualenv
```
- #create virtualenv uder the project directory
```sh
python -m venv hugging_env
```
- #activate virtual env in powershell
```sh
   .\hugging_env\Scripts\activate
```
- #install dependancy(python libraries)
```sh
pip install -r requirements.txt 
```
### 💁 Hugging Face Hub and Model Selection Process

 '''
     https://huggingface.co/
 '''
![alt text](image.png)

##### Model Memory Requirement

**ref link:** https://huggingface.co/openai/whisper-large-v3/tree/main


- #model weight
![alt text](image-1.png)


**`Memory Requirement: trained model weights * 1.2 
here, 3.09 - model weight = (3.09GB*1.2) = 3.708GB memory requires in your local pc.`**

##### Task Perform with Hugging Face

![alt text](image-2.png)

For Automatic Speech Recognation

>> Hugging Face >> Tasks >> Scroll Down >> Choose your task >> Automatic Speech Recognation >> openai
/whisper-large-v3 `[best recommendation as per Hugging Face]`

- **Pipeline code snippets to perform complex preprocessing of data and model integration.**

![alt text](image-3.png)

### 🔤 Defination of NLP

NLP 🧠💬 (Natural Language Processing) is like teaching computers to understand human language 🤖📝. It helps them read, comprehend, extract 📑🔍, translate 🌐🔤, and even generate text 📚🔍.

![alt text](image-4.png)

### 🤖 Transformers 
A Transformer 🤖🔄 is a deep learning model designed for sequential tasks, like language processing. It's used in NLP 🧠💬 for its ability to handle long-range dependencies and capture context effectively, making it crucial for tasks such as machine translation 🌐🔤, text summarization 📝🔍, and sentiment analysis 😊🔍. It's considered state-of-the-art due to its efficiency in training large-scale models and achieving impressive performance across various language tasks. 🚀📈

### 🔥 PyTorch
PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab (FAIR), featuring dynamic computational graphs and extensive deep learning support, empowering flexible and efficient model development.

### 🤗 Task-01: Chatbot
github link: 
kaggle link: 

##### 🌐 Libraries Installation

`! pip install transformers`

- 👉 model-1(blenderbot-400M-distill): https://huggingface.co/facebook/blenderbot-400M-distill/tree/main

🤖 The "blenderbot-400M-distill" model, detailed in the paper "Recipes for Building an Open-Domain Chatbot," enhances chatbot performance by emphasizing conversational skills like engagement and empathy. Through large-scale models and appropriate training data, it outperforms existing approaches in multi-turn dialogue, with code and models available for public use.

### ✨ Find Appropiate LLM Model For Specific Task
- 👉 LLM Leadears Board: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- 👉 chatbot-arena-leaderboard: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard


### 🤗 Task-02: Text Translation and Summarization

#### 🎭 Text Translation
🌐 Text Translation: Converting text from one language to another, facilitating cross-cultural communication and understanding.

##### 🌐 Libraries Installation

`! pip install transformers`
`! pip install torch`

- 👉 model-2(nllb-200-distilled-600M): https://huggingface.co/facebook/nllb-200-distilled-600M/tree/main

🌐 NLLB-200(No Language Left Behind), the distilled 600M variant, excels in machine translation research, offering single-sentence translations across 200 languages. Detailed in the accompanying paper, it's evaluated using BLEU, spBLEU, and chrF++ metrics, and trained on diverse multilingual data sources with ethical considerations in mind. While primarily for research, its application extends to improving access and education in low-resource language communities. Users should assess domain compatibility and acknowledge limitations regarding input lengths and certification.

- 👉 Language code for machine translation: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

![alt text](image-5.png)

```
### 🚩 To Clear Memory Allocation

Delete model and clear memory using **Garbage Collector**.

#garbage collector
import gc
#del model
del translator
#reclaiming memory occupied by objects that are no longer in use by the program
gc.collect()

```
#### ⛽ Text Summarization
📑 Text Summarization: Condensing a piece of text while retaining its essential meaning, enabling efficient information retrieval and comprehension.

##### 🌐 Libraries Installation

`! pip install transformers`
`! pip install torch`

- 👉 model-3(bart-large-cnn): https://huggingface.co/facebook/bart-large-cnn/tree/main


🤖 BART (large-sized model), fine-tuned on CNN Daily Mail, excels in text summarization tasks. It employs a transformer architecture with a bidirectional encoder and an autoregressive decoder, initially introduced in the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" by Lewis et al. This model variant, although lacking a specific model card from the original team, is particularly effective for generating summaries, demonstrated by its fine-tuning on CNN Daily Mail text-summary pairs.

### 🤗 Task-03: Sentence Embedding


- 🔍 Sentence Embedding Overview:
Sentence embedding represents a sentence as a dense vector in a high-dimensional space, capturing its semantic meaning.

![alt text](image-6.png)

- 🔧 Encoder Function:
During embedding, the encoder transforms the sentence into a fixed-length numerical representation by encoding its semantic information into a vector format.

- 📏 Cosine Similarity/Distance:
Cosine similarity measures the cosine of the angle between two vectors, indicating their similarity in orientation. It's vital for comparing the semantic similarity between sentences, irrespective of their magnitude.

![alt text](image-7.png)


- 🎯 Importance of Cosine Similarity/Distance:
Cosine similarity is crucial for tasks like information retrieval, document clustering, and recommendation systems, facilitating accurate assessment of semantic similarity while ignoring differences in magnitude.

##### 🌐 Libraries Installation

`! pip install transformers`
`! pip install sentence-transformers`

- 👉 model-3(all-MiniLM-L6-v2): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main

![alt text](image-8.png)

🔍 All-MiniLM-L6-v2 Overview:
The All-MiniLM-L6-v2 sentence-transformers model efficiently maps sentences and paragraphs into a 384-dimensional dense vector space, facilitating tasks such as clustering or semantic search with ease.

### 🤗 Task-04: Audio Classification

**📛Zero-Shot:**

"Zero-shot" refers to the fact that the model makes predictions without direct training on specific classes. Instead, it utilizes its understanding of general patterns learned during training on diverse data to classify samples it hasn't seen before. Thus, "zero-shot" signifies that the model doesn't require any training data for the specific classes it predicts.

- **🤖 transformers:** This library provides access to cutting-edge pre-trained models for natural language processing (NLP), enabling tasks like **`text classification, language generation, and sentiment analysis`** with ease. It streamlines model implementation and fine-tuning, fostering rapid development of NLP applications.

- **📊 datasets:** Developed by Hugging Face, this library offers a **`comprehensive collection of datasets`** for NLP tasks, simplifying data acquisition, preprocessing, and evaluation. It enhances reproducibility and facilitates experimentation by providing access to diverse datasets in various languages and domains.

- **🔊 soundfile:** With functionalities for reading and writing audio files in Python, this library enables seamless audio processing for tasks such as **`speech recognition, sound classification, and acoustic modeling`**. It empowers users to handle audio data efficiently, facilitating feature extraction and analysis.

- **🎵 librosa:** Specializing in **`music and sound analysis`**, this library provides tools for audio feature extraction, spectrogram computation, and pitch estimation. It is widely used in applications like **`music information retrieval, sound classification, and audio-based machine learning tasks`**, offering essential functionalities for audio processing projects.

#### 🔊 Ashraq/ESC50 Dataset Overview:
The Ashraq/ESC50 dataset is a collection of 2000 environmental sound recordings, categorized into 50 classes, designed for sound classification tasks. Each audio clip is 5 seconds long and represents various real-world environmental sounds, including animal vocalizations, natural phenomena, and human activities.

##### 🌐 Libraries Installation

`! pip install transformers`
`!pip install datasets`
`!pip install soundfile`
`!pip install librosa`

- 👉 model-4(clap-htsat-unfused ): https://huggingface.co/laion/clap-htsat-unfused/tree/main

![alt text](image-9.png)

🛠️ clap-htsat-unfused  offers a pipeline for contrastive language-audio pretraining, leveraging large-scale audio-text pairs from LAION-Audio-630K dataset.
The model incorporates feature fusion mechanisms and keyword-to-caption augmentation, enabling processing of variable-length audio inputs.
Evaluation across text-to-audio retrieval and audio classification tasks showcases its superior performance and availability for public use.



![alt text](image-11.png)


![alt text](image-10.png)

##### 👤 Human Speech Recording: 16,000 Hz
##### 📡 Walkie Talkie/Telephone: 8,000 Hz
##### 🔊 High Resolution Audio: 192,000 Hz

### 📌 key note for Audio Signal Processing:

For 5 sec of video, **SIGNAL VALUE** is (5 x 8000) = 40,000

In case of transformers, the **SIGNAL VALUE** relies on 🔆 Sequences and 🔆 Attention Mechanism..
**SIGNAL VALUE** will look like 60 secs for 1 secs.

In the case of transformers, particularly in natural language processing tasks, the **SIGNAL VALUE** is determined by the length of the 🔆input sequences and the 🔆 attention mechanism employed. Unlike traditional video processing, where each frame corresponds to a fixed time interval, in transformers, the **SIGNAL VALUE** may appear to be elongated due to the attention mechanism considering sequences of tokens. For example, if the attention mechanism processes 60 tokens per second, the **SIGNAL VALUE** for 1 second of input may appear equivalent to 60 seconds in terms of processing complexity.


In natural language processing, the **input sequence** refers to a series of tokens representing words or characters in a text. The **attention mechanism in transformers** helps the model focus on relevant parts of the input sequence during processing by **assigning weights to each token**, allowing the model to **prioritize important information**. Think of it like giving more attention to key words in a sentence while understanding its context, aiding in tasks like translation and summarization.

### 🤗 Task-05: Automatic Speech Recognation(ASR)

##### 🌐 Libraries Installation

`! pip install transformers`
`!pip install datasets`
`!pip install soundfile`
`!pip install librosa`
`!pip install gradio`

- !pip install transformers: Access state-of-the-art natural language processing models and tools. 🤖
- !pip install datasets: Simplify data acquisition and preprocessing for natural language processing tasks. 📊
- !pip install soundfile: Handle audio data reading and writing tasks efficiently. 🔊
- !pip install librosa: Perform advanced audio processing and analysis tasks. 🎵
- !pip install gradio: Develop interactive web-based user interfaces for machine learning models. 🌐

**Librosa** is a Python library designed for audio and music signal processing. It provides functionalities for tasks such as audio loading, feature extraction, spectrogram computation, pitch estimation, and more. Librosa is commonly used in applications such as music information retrieval, sound classification, speech recognition, and audio-based machine learning tasks.

🎙️ LibriSpeech ASR: A widely-used dataset for automatic speech recognition (ASR), containing a large collection of English speech recordings derived from audiobooks. With over 1,000 hours of labeled speech data, it facilitates training and evaluation of ASR models for transcription tasks.

👉 dataset: https://huggingface.co/datasets/librispeech_asr

👉 model: https://huggingface.co/distil-whisper

👉 model: https://github.com/huggingface/distil-whisper

**🔍 Distil-Whisper:**

Distil-Whisper, a distilled variant of Whisper, boasts 6 times faster speed, 49% smaller size, and maintains a word error rate (WER) within 1% on out-of-distribution evaluation sets. With options ranging from distil-small.en to distil-large-v2, it caters to diverse latency and resource constraints. 📈🔉

  - Virtual Assistants
  - Voice-Controlled Devices
  - Dictation Software
  - Mobile Devices
  - Edge Computing Platforms
  - Online Transcription Services

### ✨ Gradio: 

🛠️🚀 Build & Share Delightful Machine Learning Apps

Gradio offers the fastest way to showcase your machine learning model, providing a user-friendly web interface that enables anyone to utilize it from any location!

**👉 Gradio Website:** https://www.gradio.app/

**👉 Gradio In Hugging Face:** https://huggingface.co/gradio

**👉 Gradio Github:** https://github.com/gradio-app/gradio


**🌐🛠️ Gradio: Develop Machine Learning Web Apps with Ease**

Gradio, an open-source Python package, enables swift creation of demos or web apps for your ML models, APIs, or any Python function. Share your creations instantly using built-in sharing features, requiring no JavaScript, CSS, or web hosting expertise.

![alt text](image-14.png)

![alt text](image-13.png)

**📌 error:** DuplicateBlockError: At least one block in this Blocks has already been rendered.

**💉 solution:** change the `block name` that we have declared earlier.

**`demonstrations = gr.Blocks()`**

- **🚦 note:** The app will continue running unless you run **demo.close()**

### 🤗 Task-06: Text to Speech

#### Libraries Installation

`!pip install transformers`
`!pip install gradio`
`!pip install timm`
`!pip install timm`
`!pip install inflect`
`!pip install phonemizer`

- **!pip install transformers:** Installs the Transformers library, which provides state-of-the-art natural language processing models for various tasks such as text classification, translation, summarization, and question answering.

- **!pip install gradio:** Installs Gradio, a Python library that simplifies the creation of interactive web-based user interfaces for machine learning models, allowing users to interact with models via a web browser.

- **!pip install timm:** Installs Timm, a PyTorch library that offers a collection of pre-trained models and a simple interface to use them, primarily focused on computer vision tasks such as image classification and object detection.

- **!pip install inflect:** Installs Inflect, a Python library used for converting numbers to words, pluralizing and singularizing nouns, and generating ordinals and cardinals.

- **!pip install phonemizer:** Installs Phonemizer, a Python library for converting text into phonetic transcriptions, useful for tasks such as text-to-speech synthesis and linguistic analysis.

**📌Note:** py-espeak-ng is only available Linux operating systems.

To run locally in a Linux machine, follow these commands:

```sh
  sudo apt-get update
```
```sh
  sudo apt-get install espeak-ng
```
```sh
  pip install py-espeak-ng
```
**📕 APT stands for Advanced Package Tool**. It is a package management system used by various Linux distributions, including Debian and Ubuntu. APT allows users to install, update, and remove software packages on their system from repositories. It also resolves dependencies automatically, ensuring that all required dependencies for a package are installed.


- sudo apt-get update: Updates the package index of APT.
- sudo apt-get install espeak-ng: Installs the espeak-ng text-to-speech synthesizer.
- pip install py-espeak-ng: Installs the Python interface for espeak-ng.

👉 model: https://github.com/huggingface/distil-whisper



**🔍 kakao-enterprise/vits-ljs:**

🔊📚 VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

- Overview:

VITS is an end-to-end model for speech synthesis, utilizing a conditional variational autoencoder (VAE) architecture. It predicts speech waveforms based on input text sequences, incorporating a flow-based module and a stochastic duration predictor to handle variations in speech rhythm.

- Features:

🔹 The model generates spectrogram-based acoustic features using a Transformer-based text encoder and coupling layers, allowing it to capture complex linguistic patterns.

🔹 It includes a stochastic duration predictor, enabling it to synthesize speech with diverse rhythms from the same input text.

- Training and Inference:

🔹 VITS is trained with a combination of variational lower bound and adversarial training losses.

🔹 Normalizing flows are applied to enhance model expressiveness.

🔹 During inference, text encodings are up-sampled based on duration predictions and mapped into waveforms using a flow module and HiFi-GAN decoder.

- Variants and Datasets:

🔹 Two variants of VITS are trained on LJ Speech and VCTK datasets.

🔹 LJ Speech comprises 13,100 short audio clips (approx. 24 hours), while VCTK includes approximately 44,000 short audio clips from 109 native English speakers (approx. 44 hours).


👉 model: https://huggingface.co/kakao-enterprise/vits-ljs

![alt text](image-15.png)

### 📝🔊 Text to Audio Wave


Text-to-audio waveform array for speech generation is the process of converting **textual input** into a **digital audio waveform** representation. This involves `synthesizing speech from text`, where a machine learning model translates written words into spoken language. The model analyzes the text, generates corresponding speech signals, and outputs an audio waveform array that can be played back as human-like speech. The benefits include enabling natural language processing applications such as virtual assistants, audiobook narration, and automated customer service, enhancing accessibility for visually impaired individuals, and facilitating audio content creation in various industries.

**`{'audio': array([[ 0.00112925,  0.00134222,  0.00107496, ..., -0.00083117,  -0.00077596, -0.00064528]], dtype=float32), 'sampling_rate': 22050}`**

**📌note:** This dictionary contains an audio waveform represented as a NumPy array, along with its corresponding sampling rate. 🎵 The audio array consists of amplitude values sampled at a rate of 22,050 Hz.

**`{'audio': array([[ 0.00112925,  0.00134222,  0.00107496, ..., -0.00083117,  -0.00077596, -0.00064528]], dtype=float32), 'sampling_rate': 22050}`**

**📌note:** This dictionary contains an audio waveform represented as a NumPy array, along with its corresponding sampling rate. 🎵 The audio array consists of amplitude values sampled at a rate of 22,050 Hz.

### 🤗 Task-07: Object Detection

##### 🙀 Image to Audio Generation

##### 🔍 What is Object Detection


📷 Object detection is a computer vision task that involves identifying and locating objects within an image or video. The goal is to not only recognize what objects are present but also to precisely locate them by drawing bounding boxes around them. 

It's crucial for automating tasks like surveillance, autonomous driving, and quality control, enhancing safety, efficiency, and user experiences across various industries.

![alt text](image-16.png)

###🔔To Find Out the State of Art Models for Object Detection in Hagging Face

👉 Haggig Face Models: https://huggingface.co/models?sort=trending

👉 Haggig Face SoTA Models for Object Detection: https://huggingface.co/models?pipeline_tag=object-detection&sort=trending

👉 Model: https://huggingface.co/facebook/detr-resnet-50

![alt text](image-17.png)

🎊 facebook/detr-resnet-50

**DETR (End-to-End Object Detection) model with ResNet-50 backbone:**

DETR (DEtection TRansformer) model, trained on COCO 2017 dataset, is an end-to-end object detection model with ResNet-50 backbone. Utilizing encoder-decoder transformer architecture, it employs object queries for detection and bipartite matching loss for optimization, achieving accurate object localization and classification.

**📦 COCO Dataset:**

The COCO (Common Objects in Context) 2017 dataset 📷 is a widely used benchmark dataset for object detection, segmentation, and captioning tasks in computer vision. It consists of a large collection of images with complex scenes containing multiple objects in various contexts. The dataset is annotated with bounding boxes, segmentation masks, and captions for each object instance, providing rich and diverse training data for developing and evaluating object detection algorithms.

![alt text](image-18.png)

### ✨ Gradio Apps for Object Detection: 

🛠️🚀 Build & Share Delightful Machine Learning Apps For Image Genartion

Gradio offers the fastest way to showcase your machine learning model, providing a user-friendly web interface that enables anyone to utilize it from any location!

**👉 Gradio Website:** https://www.gradio.app/

**👉 Gradio In Hugging Face:** https://huggingface.co/gradio

**👉 Gradio Github:** https://github.com/gradio-app/gradio

![alt text](image-19.png)

### ✨ Make An AI powered Audio Assistant

- by importing **`summarize_predictions_natural_language`** for pipeline text generated by object detection model

### ✨ Generate Audio Narration Of An Image

- using **`kakao-enterprise/vits-ljs`**, generate text to audio

**🔍 kakao-enterprise/vits-ljs:**

🔊📚 VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

👉 model: https://huggingface.co/kakao-enterprise/vits-ljs

- Overview:

VITS is an end-to-end model for speech synthesis, utilizing a conditional variational autoencoder (VAE) architecture. It predicts speech waveforms based on input text sequences, incorporating a flow-based module and a stochastic duration predictor to handle variations in speech rhythm.


![alt text](hld-image-to-audio.jpg)


### 🤗 Task-08: Image Segmentation

Image segmentation involves dividing an image into multiple segments, each representing a distinct object or region. This process is crucial for various applications, as it simplifies image analysis and enhances understanding.

- 🏥 Medical Imaging: Used to identify tumors or anomalies in MRI or CT scans, aiding diagnosis and treatment planning.

- 🚗 Autonomous Vehicles: Enables object detection and obstacle avoidance, crucial for safe navigation on roads.

- 🌍 Satellite Imagery: Facilitates land cover classification, assisting in urban planning, agriculture, and environmental monitoring.

- In each of these examples, image segmentation plays a vital role in extracting meaningful information from complex visual data, contributing to advancements in healthcare, transportation, and environmental science.

#### Libraries Installation

`!pip install transformers`
`!pip install gradio`
`!pip install timm`
`!pip install torchvision`

`torch`

- **!pip install transformers:** Installs the Transformers library, which provides state-of-the-art natural language processing models for various tasks such as text classification, translation, summarization, and question answering.

- **!pip install gradio:** Installs Gradio, a Python library that simplifies the creation of interactive web-based user interfaces for machine learning models, allowing users to interact with models via a web browser.

- **!pip install timm:** Installs Timm, a PyTorch library that offers a collection of pre-trained models and a simple interface to use them, primarily focused on computer vision tasks such as image classification and object detection.

- **!pip install torchvision:** is used to install the torchvision library, facilitating computer vision tasks in Python environments.



👉 model: https://huggingface.co/Zigeng/SlimSAM-uniform-77

**🔍 Segmentation Anything Model (SAM)** is a versatile deep learning architecture 🧠🖼️ designed for pixel-wise segmentation tasks, capable of accurately delineating objects within images for various applications such as object detection, medical imaging, and autonomous driving.


![alt text](image-23.png)

**🔍 SlimSAM:**

SlimSAM, a novel SAM compression method, efficiently reuses pre-trained SAMs through a unified pruning-distillation framework and **employs masking for selective parameter retention**. By integrating an innovative alternate slimming strategy and a label-free pruning criterion, SlimSAM reduces parameter counts to 0.9%, MACs to 0.8%, and requires only 0.1% of the training data compared to the original SAM-H. Extensive experiments demonstrate superior performance with over 10 times less training data usage compared to other SAM compression methods.


![alt text](image-21.png)

![alt text](image-22.png)


**Masking in SlimSAM** selectively retains crucial parameters, enabling efficient compression of pre-trained SAMs without sacrificing performance, by focusing on essential features and discarding redundancies.

##### 🎨 Segmentation Mask Generation

 Segmentation mask generation involves creating pixel-wise masks that delineate different objects or regions within an image. For example, in a photo of various fruits 🍎🍌, segmentation masks would outline each fruit separately, aiding in their identification and analysis.



**💡 Key Notes:**

**`points_per_batch = 32`** in image processing denotes the number of pixel points considered in each batch during model training or inference 🖼️, aiding in efficient computation of gradients and optimization algorithms, thereby enhancing training speed and resource utilization.

**📌 note:** for smaller size in `points_per_batch` is lesser accuracy but less computationally expensive.

##### Raw image

![alt text](image-25.png)

##### Image after Segmetation:

![alt text](image-24.png)

##### 🔍 Use SlimSAM(Segment Anything Model) without Pipeline

- 🧠 The model variable initializes a SlimSAM model instance loaded from pre-trained weights 🧠🔗 located at "./models/Zigeng/SlimSAM-uniform-77", enabling tasks like inference or fine-tuning.

- 🔗 The processor variable initializes a SamProcessor instance loaded with pre-trained settings 🛠️🔗 located at "./models/Zigeng/SlimSAM-uniform-77", facilitating data preprocessing for compatibility with the SlimSAM model during inference or fine-tuning processes.

- 🛠️ Pretrained settings encompass pre-defined configurations or parameters obtained from training a model 🧠🔗, facilitating effective performance in related tasks with minimal fine-tuning or adjustment.

##### Import Libraries
`import torch`

**📌 note:** 

**no_grad** runs the model inference without tracking operations for gradient computation, thereby conserving memory resources and speeding up the inference process.

```python
with torch.no_grad(): 
  outputs = model(**inputs)
```

**Gradient computation 📈** refers to calculating the derivatives of a loss function with respect to the model parameters, crucial for updating weights during training. These gradients indicate the direction and magnitude of parameter updates needed to minimize the loss during training through optimization algorithms like gradient descent.

### 🤗 DPT


DPT (Dense Pretrained Transformer) enhances dense prediction tasks using Vision Transformer (ViT) as its backbone. It provides finer-grained predictions compared to fully-convolutional networks, yielding substantial improvements in performance, especially with large training data. DPT achieves state-of-the-art results in tasks like monocular depth estimation and semantic segmentation on datasets like ADE20K, NYUv2, KITTI, and Pascal Context.


👉 model: https://huggingface.co/docs/transformers/model_doc/dpt

👉 model in Github: https://github.com/isl-org/DPT

👉 research paper of model : https://arxiv.org/abs/2103.13413


![alt text](image-26.png)

##### Intel/dpt-hybrid-midas: 

DPT-Hybrid, also known as MiDaS 3.0, is a monocular depth estimation model based on the Dense Prediction Transformer (DPT) architecture, utilizing a Vision Transformer (ViT) backbone with additional components for enhanced performance. Trained on 1.4 million images, it offers accurate depth predictions for various applications such as autonomous navigation, augmented reality, and robotics, providing crucial depth perception for tasks like obstacle avoidance, scene understanding, and 3D reconstruction.


👉 model: https://huggingface.co/Intel/dpt-hybrid-midas

![alt text](image-27.png)

##### Demo with Gradio Apps


![alt text](image-28.png)

### 🤗 Task-09: Image to Text Retrieval 

- **🌐📸🔊 Multimodal**

Multimodal models 🌐📸🔊 are machine learning architectures designed to process and integrate information from multiple modalities, such as text, images, audio, and other data types, into a cohesive representation. These models utilize various techniques like `fusion mechanisms`, `attention mechanisms`, and `cross-modal learning` to capture rich interactions between different modalities, enabling them to perform tasks like image captioning, video understanding, and more, by leveraging the complementary information present across different modalities.

- Fusion mechanisms 🔄: Techniques to combine information from different modalities, like averaging features from text and images to make a unified representation.

- Attention mechanisms 👀: Mechanisms that focus on relevant parts of each modality's input, like attending to specific words in a sentence and regions in an image.

- Cross-modal learning 🧠💡: Learning strategies where information from one modality helps improve understanding in another, like using audio features to enhance image recognition accuracy.

**🚵 Application:** ChatGPT --> SEE, HEAR AND SPEAK

![alt text](image-29.png)

#### Bootstrapping Language-Image Pre-training🌐📸📝

BLIP Model: Proposed in BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.

**Tasks: BLIP excels in various multi-modal tasks such as:**

- Visual Question Answering 🤔➡️📸
- Image-Text Retrieval (Image-text matching) 🔎🖼️📝
- Image Captioning 🖼️📝

- Abstract: BLIP is a versatile VLP framework adept at both understanding and generation tasks. It effectively utilizes noisy web data through bootstrapping captions, resulting in state-of-the-art performance across vision-language tasks.

👉 model: https://huggingface.co/docs/transformers/model_doc/blip

👉 model: https://huggingface.co/Salesforce/blip-itm-base-coco

![alt text](image-30.png)

### 🌐💼 About Salesforce AI

Salesforce AI Research is dedicated to pioneering AI advancements to revolutionize our company, customers, and global communities 🚀. Their innovative products harness AI to enhance customer relationship management, optimize sales processes, and drive business intelligence, empowering organizations to thrive in the digital era 🌐💼.

**👉 model:** https://huggingface.co/Salesforce/blip-itm-base-coco

BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
Model card for BLIP trained on **`image-text matching`** - base architecture (with ViT base backbone) trained on COCO dataset.

![alt text](image-31.png)

#### Libraries Installation

`!pip install transformers`

`!pip install torch`

- **AutoProcessor 📊🤖** is a comprehensive tool developed by Salesforce AI Research to automate and streamline data processing tasks 🛠️. It efficiently handles data extraction, transformation, and loading processes, accelerating data-driven decision-making and improving operational efficiency across various domains such as sales, marketing, and customer service.

#### 🛠️ pt stands for pytorch

```
inputs = processor(images = raw_images,
                   text = text,
                   return_tensors = "pt"
        )
```

#### 🚴 Code Snippits: 


```
itm_scores = model(**inputs)[0]
```

- model(**inputs): This calls the model with the provided inputs. The **inputs syntax in Python unpacks the dictionary inputs and passes its contents as keyword arguments to the model function.

- [0]: This accesses the first element of the output returned by the model. The output is likely a tuple or a list containing various elements, and [0] retrieves the first element.

- itm_scores: This assigns the result obtained from step 2 to the variable itm_scores, which likely contains the predicted scores for different classes.

**📌 note:** To open a raw image

- images.jpg (image name with directory)
  

```
raw_image = Image.open("images.jpg")
raw_image
```

### 🤗 Task-10: Image Captioning 

📸🖋️ Image Captioning: Generating descriptive textual descriptions for images, enhancing accessibility and understanding of visual content.

Real-life Use: Image captioning is employed in social media platforms like Instagram to provide accessibility for visually impaired users, in content management systems for organizing and indexing images, and in educational settings for creating inclusive learning materials.

**BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**

- 🚀 BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. Utilizes noisy web data by bootstrapping captions, achieving state-of-the-art results in tasks like 📷📝 image-text retrieval and image captioning. Accessible for both conditional and unconditional image captioning.


👉 model: https://huggingface.co/Salesforce/blip-image-captioning-base
![alt text](image-32.png)

- Salesforce AI Research is dedicated to pioneering AI advancements to revolutionize our company, customers, and global communities 🚀. Their innovative products harness AI to enhance customer relationship management, optimize sales processes, and drive business intelligence, empowering organizations to thrive in the digital era 🌐💼.

- **🤖 AutoProcessor**

🤖 **`from transformers import AutoProcessor`**: Imports the AutoProcessor module from Transformers, allowing automatic loading of data processors for NLP tasks with ease. They group processing objects for text, vision, and audio modalities, providing flexibility and ease of use for various NLP tasks.

![alt text](image-33.png)


### 🎯 Image: 

![alt text](image-34.png)

### 🎯 Text Captioning after Decoding-> 

<h3>Output: two kittens in a basket with flowers</h3>

```sh
#decoding text
print(processor.decode(outputs[0], skip_special_tokens = True))
```


![alt text](image-35.png)

