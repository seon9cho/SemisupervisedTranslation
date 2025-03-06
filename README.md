# Semi-supervised Machine Translation

The current machine translation approach has mostly been a supervised task, where a bilingual dataset would be used to train an end-to-end translation system. Although these models have been shown to have fantastic performances, gathering supervised datasets is often hard and very time-consuming. With recent advances in semi/unsupervised methods, this project attempts to explore ways to train a semi-supervised machine translation system by first training language models for two separate languages and aligning them by using a few supervised examples. Although the performance of the model isn't quite perfect, I was able to show that the system can somewhat generalize to far greater data than the amount of data it was trained on.

## 1. Introduction

With the advances of neural networks, machine translation has taken giant leaps compared to previous approaches. Especially with the development of the attention mechanism and the Transformer architecture, the performance of neural machine translation systems has significantly improved. However, these systems have all been limited by the fact that they are trained using supervised data.

It is very time-consuming and requires a lot of manpower to acquire large amounts of supervised data. On the other hand, gathering unsupervised data is extremely easy in comparison. It is quite simple to scrape from domains like Reddit, Wikipedia, or Gutenberg to gather large amounts of English corpora, whereas creating translations into other languages for every sentence is a difficult task. I believe that the future of machine learning is in unsupervised algorithms, and that correct approaches using unsupervised methods will later far outperform their supervised counterpart.

In this project, I have conducted a research application stemming from the language modeling research that I have been doing over the past couple of years. I wanted to leverage the possibilities enabled by language models and create a semi-supervised machine translation model from them. Semi-supervised methods are very similar to unsupervised methods, in that most of the training of the models happens unsupervised. A few supervised examples are then used so that the models trained unsupervised can also perform tasks that require supervision.

In this project, I have trained two language models, one for English and another for French. Using these two trained models, and a very small amount of supervised data, I have attempted to build a translation system capable of translating between English and French.

## 2. Related Work

Semi/unsupervised machine translation is an active area of research. Semi/unsupervised methods are, in fact, an active area of research for many other machine learning tasks. Facebook has done extensive work in unsupervised machine translation. Their paper titled *Unsupervised Machine Translation Using Monolingual Corpora Only* was published in 2018 and showed that it is possible to train a machine translation model with no supervision whatsoever. Although the work I do in this project doesn’t necessarily bring in new techniques, it is used to show some possible applications of the language model that I’ve been researching.

## 3. Dataset

### 3.1 Data Acquisition

The dataset for this project is from the [Tatoeba project](https://www.manythings.org/anki/), a community-based project to gather machine translation data pairs for English and other languages. They have gathered bilingual language pairs for more than 100 languages to English. I chose to work with French, because it uses Latin alphabet (which is a lot easier to work with then, say, Korean), the language structure is fairly similar to English, and the amount of data available seemed reasonable.

It is important to note that the choice of language wasn't very important for the project. Ineeded to work with an easy language since the focus of the project is to develop a system that would be able to do a semi-supervised translation, hence the algorithmic design and approach was far more important than the choice of the translation language. I needed to choose one that isn't a challenge for even supervised translation, and one that is easy to process so that I don't spend too much time on solving problems that are not relevant to the project.

### 3.2 Data Splits and Details

The French-English dataset contains a total of 185,583 lines. While this isn't nearly large enough to make a high-performing translation model, it is large enough for the model to generalize to new data while also being small enough for training to be manageable. The dataset is tab-delimited, with characteristic duplicate occurrences of one language translated slightly differently in the other. For example, below are two lines found in the dataset:

<table>
  <tbody>
    <tr>
      <td>It's crowded again today.</td>
      <td>C'est de nouveau plein, aujourd'hui.</td>
    </tr>
    <tr>
      <td>It's crowded again today.</td>
      <td>Il y a de nouveau foule aujourd'hui.</td>
    </tr>
  </tbody>
</table>

This data is first split into two different files as separate English and French corpus. Each line of the two corpora are still aligned, so it is easy to generate bilingual pairs if needed. This data is then split into training and test sets with an 80-20 split, which results in 148,466 lines for the training data and 37,117 lines for the test data. Then, 2\% of the training data, which is 2970 lines, is reserved for the semi-supervised task. Throughout the paper, we will call this the 3k data.

### 3.3 Data Preprocessing

In any NLP task, data preprocessing is perhaps one of the most frustrating and time-consuming portions. To simplify the matter, I modified the data in several ways as outlined below.

#### Unicode to ASCII conversion
I first convert the text from Unicode to ASCII. This does several things. First, it eliminates any occurrences of weird Unicode characters (such as different representations of white space). These kinds of characters can add a lot of noise to the data which will hinder the performance of the model. Second, it eliminates all occurrences of accents in the letter, which are very prominent in French. This normalizes every character so that we are working with a limited alphabet. I then make sure to convert every letter to a lower case, so that the first words of sentences don't become separate vocabulary words, thus saturating the number of vocabularies in the corpus.

#### Deletion of non-alphabet characters
The next processing step is to delete all special characters and numbers. While this isn't smart when attempting to make a well-performing machine translation model, it greatly reduces the number of vocabularies, since otherwise, every number could be its own vocabulary. Processing numbers and other special characters is an important task in other aspects of NLP, but it was one that I did not want to deal with for this project. The only non-alphabet characters that would then remain are the punctuation at the end of the sentences, specifically the period, the exclamation mark, and the question mark.

#### Deletion of low-frequency vocabulary
The last step is to reduce the number of vocabulary by deleting all low-frequency vocabulary. By low-frequency, we mean words that occur only a few number of times throughout the entire corpus. For this task, I set the minimum frequency to be 5, so any word that occurs less than 5 times in the corpus will be deleted. In general, there is a logarithmic relationship between the frequency of words, and unique words with that frequency.

#### Converting to tensors
After all of the processing steps are finished, we can finally make the vocabulary for our corpus. It is necessary to convert each word into an integer (analogous to one-hot encoding) so that it can be imputed to the PyTorch Embedding module. This is simply done by creating a dictionary that maps each word to an integer, with a simple addition of PAD (padding), SOS (start of sentence), and EOS (end of sentence) tokens. A batch of sentences can then be converted into a tensor by padding them to be of the same length.

## 4 Methods

The experiments are conducted by first training two language models for each language. For the language models, I have been researching into designing an autoencoder model that takes advantage of the power of Transformer. The model encodes language in a continuous n-dimensional vector space, from which it generates an image in order to generate the output. I call this the Image Encoder Language Model, because it uses features extracted by a convolutional network from the image in order to generate it back into language. The details of model design can be found in section 4.1.

After training the language models, I then map the sentences from the 3k data in order to perform the semi-supervised training. Since the encoded representation of the two languages will reside in the same representation space, I align the distribution of these encoded points by using another transformation. The details of this process can be found in section 4.2.

### 4.1 Image Encoder Language Model
The Image Encoder Language Model is the language model that I've been working on developing in my research. The primary objective of this model is to take advantage of the power of Transformers to make an autoencoder, and the encoded representation is a vector in an n-dimensional vector space. However, because the Transformer relies on attention mechanisms to generate the output, it is insufficient to have a single vector as an input to the Transformer to generate an entire sentence. This is where image encoding comes in.

From research shown from the likes of Image Captioning, we can see that convolutional features of an image are possible to be attended on. Taking advantage of this idea, the encoded vector is first used to generate an image using upsampling techniques and convolutional kernels. The image then goes through a downsampling convolutional network that generates features from the image. These features are what are then used by the decoder transformer to generate the output.

The entire pipeline is as follows: The input sentence, at first represented as a sequence of integers, is embedded using an embedding model, and is then fed into the encoder Transformer. The output of the Transformer is of the same dimension as the input, so if represents the model dimension and represents the length of the sentence, the output of the encoder Transformer is . In order to map this into a single vector, we perform a transpose multiplication. Let A represent the output of the encoder Transformer. We then let B = A^T A, which results in a fixed-size representation. Another encoding network then processes B to obtain h, which serves as the encoded representation of the original sentence.

Now, this is then used to generate an image, and a convolutional network is used to retrieve the features out of the image. I set the number of features to be 16, so the feature vectors are what is used as memory for the decoder Transformer to generate the exact sentence as the input back out.

### 4.2 Experimental Design
Using the Image Encoder Language Model, and using the training dataset that has been split into two separate monolingual corpora, I first train the two language models for each language. One important part about this step is that both models share the same weights for the image encoder and image decoder. In order to do this, I train the French language model first. Afterwards, I use the weights of the French model for the image encoder and decoder when initializing the English model. Then, I freeze the weights of the image encoder and decoder. This way, even though both of the models were trained separately and monolingually, they will share the same weights for the image encoding and decoding.

### 4.2 Experimental Design
Using the Image Encoder Language Model, and using the training dataset that has been split into two separate monolingual corpora, I first train the two language models for each language. One important part about this step is that both models share the same weights for the image encoder and image decoder. In order to do this, I train the French language model first. Afterwards, I use the weights of the French model for the image encoder and decoder when initializing the English model. Then, I freeze the weights of the image encoder and decoder. This way, even though both of the models were trained separately and monolingually, they will share the same weights for the image encoding and decoding.

After training the two language models, I first map all of the sentences from the 3k data into the representation space. Then, I train a new model in order to align the distribution of the encoded 3k data. I do this in two different ways. I first tried doing this by purely using a linear model. This assumes that the encoded representations of the two languages are purely linear transformations of each other. Letting h_x represent the encodings of the French sentences and h_y represent the encodings of the English sentences, we would be assuming that h_y = A h_x + b for some matrix A and bias vector b. This then becomes a simple optimization problem:

\[ \min_{A, b} \| A h_x + b - h_y \| \]

Although this problem could have been approached by doing a least square regression, I decided that it would be easiest to just train a linear model in PyTorch to accomplish the same task.

The other approach was done by training a simple 3-layer neural network with ReLU activation. The hidden layer of this network was of dimension 2 times the dimension of the encoded vectors. As we’ll see in the evaluation section, this brings the loss down significantly, but doesn’t quite help that much in terms of generalization. This seems quite intuitive since with a more powerful neural network, the model can potentially transform the data points in such a way that the supervised examples would closely match, with not much regard for how the rest of the space might be changed.

---
## References

1. Vaswani, A., et al. *Attention Is All You Need*. ArXiv e-prints, June 2017.  
2. Harvard NLP. *The Annotated Transformer*. April 2018. [Link](http://nlp.seas.harvard.edu/2018/04/03/attention.html)  
3. Weng, L. *Attention? Attention!* June 2018. [Link](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms)  
4. Lample, G., et al. *Unsupervised Machine Translation Using Monolingual Corpora Only*. ArXiv e-prints, October 2017.  
5. Ott, M., et al. *Unsupervised machine translation: A novel approach to provide fast, accurate translations for more languages*. August 2018. [Link](https://engineering.fb.com/2018/08/31/ai-research/unsupervised-machine-translation-a-novel-approach-to-provide-fast-accurate-translations-for-more-languages/)  
