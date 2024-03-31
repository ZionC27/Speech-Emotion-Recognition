# Speech Emotion Recognition (SER) using deep learning

This repository contains code and resources for a Speech Emotion Recognition (SER) project, aiming to build robust models for recognizing emotions in speech signals.
The project builds upon recent studies in SER, emphasizing the significance of deep learning methods and addressing limitations in existing datasets.

You can test out the Speech Emotion Recognition on my hugging face spaces here: https://huggingface.co/spaces/ZionC27/Speech-Emotion-Recognition

Dataset Description and Analysis:
A comprehensive dataset was constructed by combining secondary datasets including Emotional Multimodal Actors Dataset (CREMA-D), JL corpus, Toronto Emotional Speech Set (TESS), EmoV- DB, ASVP-ESD (Speech and Non-Speech Emotional Sound), 
Publicly Available Emotional Speech Dataset (ESD), Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), and a primary dataset Diverse Emotion Speech dataset - English (DESD-E) collected from friends and schoolmates. 
This approach ensures diversity and richness in the dataset, contributing to the robustness of the emotion recognition models. 
The decision to incorporate primary data stemmed from limitations observed in existing datasets, including a focus on specific sentences and accent variations.

# Feature Extraction Methods

Zero-Crossing Rate (ZCR): ZCR calculates the rate at which the audio signal changes its sign, providing insights into speech characteristics such as speech rate and energy distribution.

Root Mean Square (RMS): RMS quantifies the overall energy present in the speech signal, offering valuable information about speech intensity and loudness variations.

Mel Frequency Cepstrum Coefficient (MFCC): MFCC captures the spectral envelope of the speech signal, emphasizing perceptually relevant features related to speech timbre, pitch, and spectral shape.

# Model

The project utilizes deep learning techniques for emotion classification, incorporating Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) such as Long Short-Term Memory (LSTM), Bidirectional Long Short-Term Memory (Bi-LSTM)
and Gated Recurrent Units (GRUs). Various machine learning models are explored, including different LSTM variants and combinations of CNN and LSTM architectures. 
Among the experimented models, the CLSTM (CNN + LSTM) architecture emerges as the top performer, achieving an impressive accuracy of 82.12% and a precision of 84.66%. This model effectively integrates CNN for spatial feature extraction and 
LSTM for temporal dependency modeling, allowing it to capture intricate patterns in the speech data. The model can be found here: https://huggingface.co/ZionC27/EMO_20_82

# Future Work

Collecting More Data for DES-D: Efforts will be directed toward expanding the private dataset used in this project. Additional data will be collected to augment the existing dataset, 
ensuring better coverage of diverse emotional expressions and linguistic variations. This expanded dataset will contribute to enhancing the robustness and effectiveness of the emotion recognition models.

Model Fine-Tuning: Further refinement of the SER models will involve fine-tuning of hyperparameters and architecture adjustments. This iterative process aims to optimize model performance and improve accuracy in emotion classification tasks.

Exploring More Emotions: The inclusion of additional emotional categories beyond the existing ones will be explored. This expansion will enable the SER system to recognize a wider spectrum of emotions, enhancing its capability to capture the nuances of human emotion.

Incorporating Different Languages: Efforts will be made to incorporate speech samples from different languages into the dataset. Training and evaluating the models on multilingual data will enable emotion 
recognition in diverse linguistic contexts, expanding the applicability of the SER system.

## Setup

Python: version 3.9 and above should work

Required libraries:
```
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install tensorflow
pip install librosa
pip install tensorflow
pip install keras-tuner
```
All audio file used for testing and training should be in Wav format

For any questions or inquiries, please contact me at zion2027@gmail.com
