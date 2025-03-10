# Semi-supervised Machine Translation

The prevailing approach to machine translation has primarily been a supervised task, where a bilingual dataset is used to train an end-to-end translation system. While these models have demonstrated impressive performance, obtaining high-quality supervised datasets is often challenging and time-consuming. With recent advancements in semi-supervised and unsupervised learning methods, this project aims to explore training a semi-supervised machine translation system by first independently training language models for two different languages and then aligning them using a small number of supervised examples. Although the model's performance isn't the best, the results demonstrate that the system can generalize to significantly larger datasets than those it was originally trained on.

## 1. Introduction

With the advancement of neural networks, machine translation has made remarkable progress compared to traditional approaches. In particular, the development of the attention mechanism and the Transformer architecture has significantly enhanced the performance of neural machine translation (NMT) systems, surpassing previous methods by a wide margin. However, despite these advances, most NMT systems still rely heavily on supervised data for training. Obtaining large, high-quality supervised datasets is often labor-intensive, time-consuming, and costly. In contrast, acquiring unsupervised data is comparatively easy — large corpora of text in a single language can be readily obtained from sources such as Reddit, Wikipedia, or Project Gutenberg. However, creating corresponding translations for every sentence in another language remains a challenging task.

I believe that the future of machine learning lies in unsupervised algorithms, and that with the right approach, unsupervised methods could eventually outperform their supervised counterparts. This project builds on my prior research in language modeling and aims to explore semi-supervised machine translation. The core idea is to leverage pre-trained language models and introduce a small amount of supervised data to guide the model in performing translation tasks. Semi-supervised methods closely resemble unsupervised methods in that most of the model's training occurs without supervision, with only a limited amount of labeled data used to steer the model toward the desired task.

In this project, I trained two separate language models — one for English and one for French. By aligning these models using a minimal amount of supervised data, I attempted to build a translation system capable of translating between English and French. Although the resulting model is not yet perfect, the findings demonstrate the potential of semi-supervised methods in machine translation, particularly in scenarios where supervised data is scarce.

## 2. Related Work

Semi-supervised and unsupervised machine translation is an active and rapidly evolving area of research, with broader implications across many other machine learning tasks. In particular, unsupervised methods have gained significant attention for their ability to reduce reliance on labeled data. One notable example is Facebook's 2018 paper titled Unsupervised Machine Translation Using Monolingual Corpora Only, which demonstrated that it is possible to train a machine translation model entirely without supervision. Although the work presented in this project does not introduce new techniques, it serves as a practical demonstration of how the language models I have been developing can be applied to semi-supervised machine translation.

## 3. Dataset

### 3.1 Data Acquisition

The dataset used for this project comes from the [Tatoeba project](https://www.manythings.org/anki/), a community-driven initiative aimed at collecting bilingual sentence pairs for machine translation across over 100 languages. For this project, I chose to work with French due to several practical considerations. French uses the Latin alphabet, making it easier to process than languages with non-Latin scripts such as Korean. Additionally, the grammatical structure of French is relatively similar to English, and the dataset size was large enough to be useful without being overwhelming.

It is worth noting that the choice of language was not a critical factor for this project. Since the primary goal was to develop a semi-supervised machine translation system, the focus was on the algorithmic design rather than the complexity of the target language. I deliberately chose a language that posed minimal challenges for standard supervised translation and was easy to preprocess, allowing me to allocate more time to refining the model itself rather than addressing language-specific complexities.

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

The data is first separated into two files, creating distinct English and French corpora. Despite the separation, the lines remain perfectly aligned, allowing for easy generation of bilingual pairs when needed. The dataset is then split into training and test sets using an 80-20 ratio, resulting in 148,466 lines for training and 37,117 lines for testing. Additionally, 2% of the training data — approximately 2,970 lines — is set aside for the semi-supervised task. This subset will be referred to throughout the paper as the _3k data_.

### 3.3 Data Preprocessing

In any natural language processing (NLP) task, data preprocessing is often one of the most tedious and time-consuming steps. To simplify this process and ensure cleaner input for the model, I applied several preprocessing techniques, as outlined below.

#### Unicode to ASCII Conversion
The first step was to convert text from Unicode to ASCII. This served two main purposes. First, it eliminated any occurrences of unusual Unicode characters (such as different representations of whitespace), which can introduce noise and hinder model performance. Second, it stripped accents from characters — a common feature in French — thereby normalizing the text and reducing variability in the dataset. Additionally, I converted all text to lowercase to prevent the model from treating capitalized words (such as the first word of a sentence) as distinct vocabulary, which could unnecessarily inflate the vocabulary size. This step helped standardize the text and simplify model training.

#### Removal of Non-Alphabetic Characters
The next step was to remove all special characters and numbers from the text. While preserving such characters is generally beneficial for building robust machine translation models, I chose to discard them to simplify the task and reduce the vocabulary size. Without this step, every unique number or special character could become its own vocabulary token, unnecessarily complicating the model. However, I did retain basic punctuation marks — specifically the period, exclamation mark, and question mark — as they provide important structural cues for sentence boundaries.

#### Removal of Low-Frequency Vocabulary
To further reduce the vocabulary size, I removed all low-frequency words from the corpus. Low-frequency words, defined in this project as those appearing fewer than five times in the dataset, contribute little to model learning but significantly increase the vocabulary size and computational complexity. By setting a frequency threshold of five, I ensured that only words appearing with reasonable regularity were retained. This step is particularly effective given the logarithmic relationship between word frequency and vocabulary size in natural language data, allowing the model to focus on more representative vocabulary.

#### Converting Text to Tensors
After completing the preprocessing steps, the final task was to convert the text into a numerical format suitable for model input. This involved creating a vocabulary dictionary that mapped each word to a unique integer, along with three special tokens: PAD (padding), SOS (start of sentence), and EOS (end of sentence). This approach is analogous to one-hot encoding, enabling the text to be passed to PyTorch's Embedding module. To efficiently process batches of text, I padded all sentences to a uniform length, allowing them to be stacked into tensors for training.

## 4 Methods

The experiments begin by training separate language models for each language. For these models, I utilized an autoencoder architecture that I have been researching, which leverages the power of the Transformer. The model encodes input text into a continuous n-dimensional vector space, from which it reconstructs the output. Uniquely, my model incorporates an image generation step during training, where it generates an image representation from the encoded text and then uses features extracted from the image — via a convolutional network — to reconstruct the original text. I refer to this architecture as the _Image Encoder Language Model_. Further details regarding the model design can be found in Section 4.1.

Once the language models are trained, I perform semi-supervised training using the _3k data_. Since the encoded representations of the two languages exist within the same continuous vector space, I apply an additional transformation to align their distributions. This alignment enables the model to translate between languages despite being trained with minimal paired data. The specific details of this alignment process are described in Section 4.2.

### 4.1 Image Encoder Language Model
The Image Encoder Language Model is the language model I have been developing in my research. Its primary objective is to leverage the power of Transformers to create an autoencoder, where the encoded representation exists as a vector in an n-dimensional vector space. However, since the Transformer relies heavily on the attention mechanism to generate outputs, providing a single vector as input is insufficient to produce an entire sentence. This is where image encoding becomes useful.

Drawing inspiration from research in Image Captioning, where convolutional features of an image can be effectively attended to, I adopt a similar approach. The encoded vector is first used to generate an image through upsampling techniques and convolutional kernels. This generated image is then processed through a downsampling convolutional network to extract meaningful features. These extracted features are subsequently used by the decoder Transformer to generate the output text. This approach enables the model to effectively utilize the information captured in the encoded vector through visual representations.

The entire pipeline operates as follows: The input sentence, initially represented as a sequence of integers, is first embedded using an embedding model and then fed into the encoder Transformer. The output of the Transformer has the same dimension as the input, meaning that if $N$ denotes the model dimension and $T_x$ denotes the length of the sentence, the output of the encoder Transformer is of size $T_x \times N$. To map this output into a single vector, we perform a transpose multiplication. Let $A$ denote the output of the encoder Transformer, such that $A \in \mathbb{R}^{T_x \times N}$. We then compute $B = A^T A$, resulting in $B \in \mathbb{R}^{N \times N}$. Notice that this operation eliminates the variable-length dimension of the input sentence. Subsequently, another encoding network processes $B$ to obtain $h \in \mathbb{R}^{N}$, which serves as the fixed-size encoded representation of the original sentence.

The vector $h$ is then used to generate an image, which is subsequently processed by a convolutional network to extract features. I set the number of extracted features to 16, resulting in a $16 \times N$ feature map. This feature map acts as the memory for the decoder Transformer, enabling it to reconstruct the original input sentence. By doing so, the model captures and utilizes the encoded information in a structured and visual form, allowing the decoder to attend to relevant features during sentence reconstruction.

### 4.2 Experimental Design
Using the Image Encoder Language Model and a training dataset split into two separate monolingual corpora, I first train a language model for each language. A key aspect of this step is ensuring that both models share the same weights for the image encoder and decoder. To achieve this, I begin by training the French language model. Once trained, I use the weights from the French model to initialize the image encoder and decoder for the English model. During the training of the English model, I freeze the weights of the image encoder and decoder. This approach ensures that, despite being trained separately and monolingually, both models share identical weights for image encoding and decoding.

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
