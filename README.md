# Chinese2English-Translation-seq2seq
Build you own translator from chinese to english with seq2seq model in pytorch😄

## Requirements
- `torchtext >= 1.8'
- `spacy`
## Structure
### Files
- `get_loaderl.py`: Define the dataloader using torchtext
- `train.py`: Train the model
- `model.py`: Define the model
- `inference.py`:Translate your own chinese sentece to english one !!
### Folders
- 'saved_vocab`: Contain serval vocabulary txt and you can also generate then during training
- `translation2019zh`: This is Google's chinese2english translation samples. It's huge and i only take the validation dataset to train

## How to use
### How to train
- 'Go inside the train.py, set some hyperparameters if you want or just run it!'
- 
### How to translate my own sentence
- `Go inside the inference.py, set the your own chinese sentence at line 73 

## Contact me for trained_weights(too big to upload)
- mountchicken@outlook.com
