<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Mentor][mentor-shield]][mentor-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# Neural Embeddings
________

## Link to Heroku Web page for Sentiment Analysis and Multi-class question type
- https://neural-eyes.herokuapp.com/

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Tokenization](#Tokenization)
* [Why Neural Embeddings Are Preferred](#Why-Neural-Embeddings-Are-Preferred)
* [Word2vec : CBOW And Skipgram](#Wordvec-Cbow-And-Skipgram)
* [Colab Notebook References](#Colab-Notebook-References)
* [Model Weight References For Future Training](#Model-Weight-References-For-Future-Training)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Linux](https://www.tutorialspoint.com/ubuntu/index.htm)
* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.5.0](https://pytorch.org/) ** Higher versions may cause slugspace issues while deploying to Heroku 
* [torchtext 0.6.0](https://pytorch.org/docs/stable/torchvision/index.html) ** Higher versions may cause slugspace  issues while deploying to Heroku
* [Google Colab](https://colab.research.google.com/)
* [Spacy 2.3.2](https://spacy.io/)
* [en-core-web-sm](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz#egg=en_core_web_sm==2.3.1)

<!-- Tokenization -->
## Tokenization
#### How to feed text data to neural networks
- Neural networks only understand numbers. Hence like images, text also needs to be converted to numbers to feed to neural networks.
- We can call this vectorization i.e. converting text into vectors. 
- There are mainly 2 parts involved in vectorization of text - tokenization and numericalization.
- Tokenization means creating tokens out of a given sentence. Example as below:
  "This is Anil Bhatt !" --> ["This", "is", "Anil", "Bhatt", "!"]
- Numericalization means assigning an index for these numbers based on their position in vocabulary. Example as below:
  ["This", "is", "Anil", "Bhatt", "!"] --> [9876, 5346, 567, 1320, 43]
- From where will we get vocabulary ? Building such a vocabulary will be cumbersome. Hence, We usually use standard word vectors like Glove for this purpose.
  Example : "glove.6b.100d" --> Means Glove trained on 6 Billion words and having 100 dimensions.
- Tokenization is also not straight forward process if we choose to build it. Example : "Stock value crashed by $100B for Apple over nite :( ...(sic)" . We 
 can see that there are several words like 100(can be any number), B(here it denotes Billion but can mean different things based on context), sic(means  
 quoted verbatim, hence cannot put under unknown word category), :( (sad smiley, not some random characters).
- Tokenization of above sentence will not be easy. Hence, for tokenization also, we will use standard tokenizers like Spacy.  
- These tokenized inputs are passed to an embedding layer which will give a vector representation of the tokenized word based on its value in embedding   
  layer. Imagine embedding layer as a look-up table. 
  Text Input to embedding layer shape = [batch size, sentence len], Output from embedding layer shape = [batch size, sentence len, embedding dimension]
- This embedding layer output will then be fed to subsequent DNN layers to predict an output. Output could be next word in sequence, sentiment (+ve or -ve as 
  in IMDB movie review dataset), type of question (HUM, ETY, DESC etc as in TREC dataset) etc.
- Tokenization we referred above is word-by-word tokenization which is the most popular one. We can also use characted-by-characted tokenization. Another 
  method is n-grams. An example of bigram is as follows: "My name is Anil !" --> bigram will be "My name", "name is", "is Anil", "Anil !"
- However, use of convolutions eliminates need for using n-grams. Please refer below code bases to understand how convolutions eliminate need for n-grams.
  https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Faster_Sentiment_Analysis_using_FastText.ipynb
  https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Convolutional_Sentiment_Analysis.ipynb

<!-- Why-Neural-Embeddings-Are-Preferred -->
## Why Neural Embeddings Are Preferred
- There are multiple ways to associate a vector with word - one is one-hot encoding, another is word embeddings.
- Problem with one-hot encoding that it will return a sparse high-dimensional vector which is expensive. Example : Let us say our vocabulary has 100 words 
  and "Therefore" is one among these words. Let us say a sentence having "Therefore" comes up for vectorization. In this case, token for "Therefore" will be 
  a 100 element vector with index corresponding to "Therefore" only having a value of 1 and rest 99 indexes as zeroes.
- As we can clearly see, this is wasteful. Thus came the idea of neural word embeddings.
- Unlike the word vectors obtained via one-hot encoding, word embeddings are learned from data.
- Below image clearly depicts difference between these two.

![One_hot vs Word_Embedding](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/One_hot%20vs%20Word%20Embedding.jpg)

<!-- Wordvec-Cbow-And-Skipgram -->
## Word2Vec:CBOW and Skipgram
- Word2Vec is a shallow, two-layer neural network which is trained to reconstruct linguistic contexts of words.
- It comes in two flavors, the Continuous Bag-of-Words (CBOW) model and the Skip-Gram model.
- Algorithmically both are similar as shown below:
![Algorithm View](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/CBOW_SkipGram.png)
- Example of CBOW vs Skip-Gram is as below:
![Example](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/CBOW_Example.png)

<!-- Colab-Notebook-References -->
## Colab Notebook References
- Below ipynb notebooks will help us to get a better understanding on NLP models. Details are as below.
- Original source of these notebooks https://github.com/bentrevett/pytorch-sentiment-analysis
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Simple_Sentiment_Analysis.ipynb
- Task : Predict sentiment (+ve 1 or -ve 0) from IMDB movie reviews
- Tokenizer : Spacy. Also, we are building vocabulary ourselves.
- We define a simple embedding layer here. Architecture is embedding -> RNN -> fc -> Prediction
- As it is a binary classification problem, we are using nn.BCEWithLogitsLoss() as loss function
- Model randomly predicts sentiments randomly (Val. Acc: 51.03%) indicating that it is not a useful model. Trained for 20 epochs.
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Updated_Sentiment_Analysis.ipynb
- Here we are improving upon previous version
- Task : Predict sentiment (+ve 1 or -ve 0) from IMDB movie reviews
- Tokenizer : Spacy with pre-padded sequencing, uses LSTM instead of RNN, also uses bidirectional & multi-layer concepts of RNNs in model.
- Built vocabulary from glove.6B.100D and pre-trained glove embedding is used. As dataset is large only top 25000 words are used for building vocab, rest are treated as <unk>.
- As it is a binary classification problem, we are using nn.BCEWithLogitsLoss() as loss function
- Model performance improves significantly with Val. Acc: 89.62%. Trained for 20 epochs.
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Faster_Sentiment_Analysis_using_FastText.ipynb
- Task : Predict sentiment (+ve 1 or -ve 0) from IMDB movie reviews
- Tokenizer : Spacy. FastText model used which employs n-grams. Here we used bigrams. As RNNs are not involved, not using pre-padded sequences.
- Built vocabulary from glove.6B.100D and pre-trained glove embedding is used. As dataset is large only top 25000 words are used for building vocab, rest are treated as <unk>.
- FastText architecture is embedding -> Average Pooling -> FC -> Prediction
- As it is a binary classification problem, we are using nn.BCEWithLogitsLoss() as loss function
- Model performance further improves with Val. Acc: 90.74%. Trained for 20 epochs.
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Convolutional_Sentiment_Analysis.ipynb
- Task : Predict sentiment (between 0 & 1) from IMDB movie reviews
- Tokenizer : Spacy. Built vocabulary from glove.6B.100D and pre-trained glove embedding is used. As dataset is large only top 25000 words are used for building vocab, rest are treated as <unk>.
- CNN model is used. We decide n-grams using different filter sizes. Eg: filter_size = 3, tri-grams used
- Architecture is as depicted below.
![CNN](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/CNN.jpg)
- As it is a binary classification problem, we are using nn.BCEWithLogitsLoss() as loss function
- Model performance slightly drops with Val. Acc: 88.09% but processing becomes much faster. Trained for 20 epochs.
- Refer for CPU models & vocab pkl file creation for future training/edge deployment.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Convolutional_Sentiment_Analysis_cpu.ipynb
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Multi_Class_Question_Type_Analysis.ipynb
- Task : TREC dataset used here. This is a dataset of questions and task is to classify what category question belongs to. eg: HUM for questions about humans
- Total 6 category of questions are present, hence number of classes = 6
- Tokenizer : Spacy. Built vocabulary from glove.6B.100D and pre-trained glove embedding is used. Dataset is small with only ~7500 unique words. Hence no restrictions in the form of top words is not employed (as we have seen with IMDB review).
- Same CNN model as earlier is used. 
- As it is a multi classification problem, we are using nn.CrossEntropyLoss() as loss function
- Model performance is good with Val. Acc: 83.97% when trained for 20 epochs
- Refer for CPU models & vocab pkl file creation for future training/edge deployment.
https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Multi_Class_Question_Type_Analysis_cpu.ipynb
### https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Transformer_Senti_Analysis.ipynb
- Task : Predict sentiment (+ve 1 or -ve 0) from IMDB movie reviews
- We are using transformers here. Model/tokenizer used is BERT with no casing.
- Glove is not used for embedding as we need to use same vocabulary as in transformer. 
- Model used is GRU. Architecture is embedded(bert) -> GRU -> FC -> Prediction
- As it is a binary classification problem, we are using nn.BCEWithLogitsLoss() as loss function
- Processing takes significant amount of time ~ 18 minutes for 1 epoch. Accuracy also jumps up Val. Acc: 92.52% trained for 5 epochs

<!-- Model-Weight-References-For-Future-Training -->
## Model Weight References For Future Training
- Refer below locations to download pretrained weights for future. GPU versions(for colab training) are available for all versions. CPU versions are available for Convolutional sentiment analysis and multi-class question type models.
https://drive.google.com/drive/folders/1Vmb34RHxjtKf19HmfMfwO3sjri_XwY4u?usp=sharings

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MENTOR -->
## Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) , [The School of A.I.](https://theschoolof.ai/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[mentor-shield]: https://img.shields.io/badge/Mentor-mentor-yellowgreen
[mentor-url]: https://www.linkedin.com/in/rohanshravan/
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555



