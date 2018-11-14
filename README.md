# PyTorch Implementation of NLU Generative Models #


## Features

 - **train**: train a generative model using a labeled NLU dataset.
 - **predict**: predict intents and slot labels for an unlabeled NLU dataset.
 - **generate**: generate utterances and their respective intents and slot labels with the given latent variable sampling distribution (uniform, gaussian, gaussian mixture, etc.)

Run `python -m (train|predict|generate) --help` to checkout available options
 
 
## Required Packages

Python >3.6 is required at the least.

 - Install packages listed in `requirements.txt`.
 - To validate generated sentences using the Universal Sentence Encoder (arxiv:1803:11175), install `tensorflow` and `tensorflow-hub` packages.
 - To use tensorboard as the visualization tool, install `tensorboardX==1.7`
 