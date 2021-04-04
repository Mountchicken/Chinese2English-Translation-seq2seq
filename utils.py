import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, chinese, english, device, max_length=50):
    # Load chinese tokenizer
    spacy_ch= spacy.load("zh_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ch(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, chinese.init_token)
    tokens.append(chinese.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [chinese.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, chinese, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["ch"]
        trg = vars(example)["eng"]

        prediction = translate_sentence(model, src, chinese, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def save_vocab(vocab,save_path):
    with open(save_path,'w',encoding='utf-8') as f:
        f.write(str(vocab))

def load_stoi(save_path):
    with open(save_path,'r',encoding='utf-8') as f:
        dic=f.read()
        dic=list(dic)
        for i in range(len(dic)):
            if dic[i]=='{':
                del dic[0:i]
                del dic[-1]
                break
        dic=''.join(dic)
        dic=eval(dic)
    return dic

def load_itos(save_path):
    with open(save_path,'r',encoding='utf-8') as f:
        dic=f.read()
        dic=eval(dic)
    return dic

