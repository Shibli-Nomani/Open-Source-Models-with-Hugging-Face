# Open Source Models with Hugging Face
- **👉 chatbot code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/chatbot.ipynb
- **👉 text translation and text summarization code:** Open-Source-Models-with-Hugging-Face/notebooks
/text translation and text summarization.ipynb
- **👉 sentence embedding code:** https://github.com/Shibli-Nomani/Open-Source-Models-with-Hugging-Face/blob/main/notebooks/Sentence%20Embeddings.ipynb
- 
**🤗 Hugging Face Overview:**
Hugging Face is a leading platform for natural language processing (NLP), offering a vast repository of pre-trained models, datasets, and tools, empowering developers and researchers to build innovative NLP applications with ease.

![alt text](<Hugging Face.png>)

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

