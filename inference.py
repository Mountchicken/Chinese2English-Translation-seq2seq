import torch
import spacy
from utils import load_stoi, load_itos
from model import Encoder, Decoder, Seq2Seq
def translate(model, sentence, chinese_vocab, english_vocab, device, max_length=50):
    # Load chinese tokenizer
    spacy_ch= spacy.load("zh_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ch(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Go through each german token and convert to an index
    text_to_indices = [chinese_vocab['stoi'][token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english_vocab['stoi']["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english_vocab['stoi']["<eos>"]:
            break

    translated_sentence = [english_vocab['itos'][idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

if __name__=="__main__":
    #load vocab
    chinese_itos = load_itos('saved_vocab/chinese_itos.txt')
    english_itos = load_itos('saved_vocab/english_itos.txt')
    chinese_stoi = load_stoi('saved_vocab/chinese_stoi.txt')
    english_stoi = load_stoi('saved_vocab/english_stoi.txt')
    chinese_vocab={'stoi':chinese_stoi,'itos':chinese_itos}
    english_vocab={'stoi':english_stoi,'itos':english_itos}

    # Model hyperparamters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size_encoder = len(chinese_stoi)
    input_size_decoder = len(english_stoi)
    output_size = len(english_stoi)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5

    #initialize model
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    model = Seq2Seq(encoder_net, decoder_net, len(english_stoi)).to(device)

    param = torch.load('my_checkpoint.pth.tar')["state_dict"]
    model.load_state_dict(param)
    model.eval()

    #test sentence
    sentence = '昨天有人去超市买了一瓶啤酒'
    trans = translate(model, sentence, chinese_vocab, english_vocab, device, max_length=50)
    print(trans)