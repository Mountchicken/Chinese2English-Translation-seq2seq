#from torchtext.data import Field, BucketIterator, TabularDataset # for torchtext 0.8
from torchtext.legacy.data import Field, BucketIterator, TabularDataset # for torchtext 0.98
import numpy as np
import spacy
from utils import save_vocab

#Define tokenizer
spacy_ch= spacy.load("zh_core_web_sm")
spacy_eng= spacy.load("en_core_web_sm")

def tokenize_ch(text):
    return [tok.text for tok in spacy_ch.tokenizer(text)]
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def get_loader(batch_size, save_vocabulary=False):
    english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True,init_token='<sos>',eos_token='<eos>')
    chinese = Field(sequential=True, use_vocab=True, tokenize=tokenize_ch,lower=True,init_token='<sos>',eos_token='<eos>')

    fields = {"english":("eng", english), "chinese":("ch", chinese)}

    train_data, test_data = TabularDataset.splits(
        path="translation2019zh/",
        train="translation2019zh_valid.json",
        test="translation2019zh_valid.json",
        format="json",
        fields=fields
    )

    english.build_vocab(train_data, max_size=30000, min_freq=2)
    chinese.build_vocab(train_data, max_size=30000, min_freq=2)
    
    if save_vocabulary:
        save_vocab(english.vocab.stoi, 'saved_vocab/english_stoi.txt')
        save_vocab(english.vocab.itos, 'saved_vocab/english_itos.txt')
        save_vocab(chinese.vocab.stoi, 'saved_vocab/chinese_stoi.txt')
        save_vocab(chinese.vocab.itos, 'saved_vocab/chinese_itos.txt')

    train_iterator,_ = BucketIterator.splits(
        (train_data,test_data),
        batch_size=batch_size,
        device="cuda",
        sort_within_batch= True,   #按照句子的长短来构成batch，减小padding的计算量
        sort_key = lambda x :len(x.ch)
    )
    return train_iterator, english, chinese