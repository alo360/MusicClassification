# Music Genre Classification
Introduction
------------

Music is categorized into subjective categories called genres. With the growth of the internet and multimedia systems applications that deal with the musical databases gained importance and demand for Music Information Retrieval (MIR) applications increased. Musical genres have no strict definitions and boundaries as they arise through a complex interaction between the public, marketing, historical, and cultural factors. This is small demo that Classify Music in to genres. 

Requirements
------------

* Streamlit
* Tensorflow
* Numpy 
* Scikit-Learn 
* Scipy 
* Python-Speech-Features 
* Pydub 
* Librosa 


## Music Genre Classifier model
Our demo application is written in Python using streamlit lib. It uses a trained `CNN model` for finding the genre. 
Using keras layers of Conv2D, MaxPool2D, BatchNormalization.

CNN layers takes input primarily in 3D shape, so we again have to prepare the dataset in the form and for that, I have used np.newaxis function which adds a column/layer in the data

-----------

We need to find the best classification algorithm that can be used in our Web App. Matlab is ideal to implement machine learning algorithms in minimum lines of code. Before making the Web App in python we made a prototype in Matlab. 

## Feature Extraction                                            
Each genre comprises 100 audio files (.wav) of 30 seconds each that means I have 1000 training examples and if I keep 20% of them for validation then just 800 training examples.
I chose to extract MFCC from the audio files as the feature. For MFCC feature, I have used librosa.feature.mfcc function of librosa. The output will be a matrix of 13*n dimensional vector. Where n depends on the total duration of the audio. 13*(100*sec).
* SAMPLE_RATE = 22050
* DURATION = 30
Which the one file, I has collect feature for 30 second and split into 10 samples. This means that one samples nearly 3 seconds. Now my training examples have become tenfold i.e. each genre has 1000 training examples and total training examples are 10,000. So we increased my dataset and this will be helpful for a deep learning model because it always requires more data.
The normal sampling rate is 22050 Hz. If you resampled your signal at a higher rate, you would likely not have any detectable difference in the output.

### Mel-Frequency Cepstral Coefficients (MFCC)
MFCC is that one feature you would see being used in any machine learning experiment involving audio files.
Generally, the first 13 coefficients(the lower dimensions) of MFCC are taken as features as they represent the envelope of spectra. And the discarded higher dimensions express the spectral details. For different phonemes, envelopes are enough to represent the difference, so we can recognize phonemes through MFCC. 
* (MFCCs) of a signal are a small set of features (usually about 10-20) which concisely describe the overall shape of a spectral envelope. In MIR, it is often used to describe timbre.
  
## Classification
### Convolutional Neural Network (CNN)
CNN
Using keras layers of Conv2D, MaxPool2D, BatchNormalization.

CNN layers takes input primarily in 3D shape, so we again have to prepare the dataset in the form and for that, I have used np.newaxis function which adds a column/layer in the data

|Layer (type)                 | Output Shape            | Param #|
| --------                    | --------                | -------|
|conv2d (Conv2D)              | (None, 128, 11, 64)     | 640    |
|max_pooling2d (MaxPooling2D) |(None, 64, 6, 64)        | 0      |
|batch_normalization (BatchNo |(None, 64, 6, 64)        | 256    |
|conv2d_1 (Conv2D)            |None, 62, 4, 32)         | 18464  |
|max_pooling2d_1 (MaxPooling2 |None, 31, 2, 32)         | 0      |
|batch_normalization_1 (Batch |None, 31, 2, 32)         | 128    |
|conv2d_2 (Conv2D)            |None, 30, 1, 32)         | 4128   |
|max_pooling2d_2 (MaxPooling2 |None, 15, 1, 32)         | 0      |
|batch_normalization_2 (Batch |None, 15, 1, 32)         | 128    |
|conv2d_3 (Conv2D)            |None, 15, 1, 16)         | 528    |
|max_pooling2d_3 (MaxPooling2 |None, 8, 1, 16)          | 0      |
|batch_normalization_3 (Batch |None, 8, 1, 16)          | 64     |
|flatten_2 (Flatten)          |None, 128)               | 0      |
|dense_9 (Dense)              |(None, 64)               | 256    |
|dropout_3 (Dropout)          |(None, 64)               | 0      |
|dense_10 (Dense)             |None, 10)                | 650    |
|Total params: |33,242|
|Trainable params: |32,954|
|Non-trainable params: |288|


Python package 
--------------------

```
????????? class_xls
???   ????????? model_cnn_weights.h5
???   ????????? model_cnn.json
????????? genres_original
???   ????????? 
???   ????????? 
???   ????????? 
???   ????????? ......
????????? codeFeature
???   ?????????  feature_extractor_cnn.py
???   ?????????  file_split.py
????????? main.py
```
### *codeFeature* 
This module is used to extract MFCC features from a given file. It contains the following functions.
* ***class_xls (file):*** 
Extract features from a given file. Given the model and weight of CNN.
* ***genres_original (audio_dir):*** 
All dataset files in a directory. 

Results
=======


## Conclusion

