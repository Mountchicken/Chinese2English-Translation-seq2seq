import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import Encoder, Decoder, Seq2Seq

def train():
    # Traing
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64

    #get trainloader and vocab
    save_vocab = True #是否保存词典，建议第一次训练的时候保存，后面就不需要保存了
    train_iterator, english, chinese = get_loader(batch_size, save_vocab)

    # Model hyperparamters
    load_model = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size_encoder = len(chinese.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5

    #Tensor board
    writer = SummaryWriter(f'runs/loss_plot')
    step = 0


    # Initialize network
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    model = Seq2Seq(encoder_net, decoder_net, len(english.vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pad_idx = english.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'),model, optimizer)
    sentence = "你知道的，我会永远爱着你。" #测试用句
    for epoch in range(num_epochs):
        batch = next(iter(train_iterator))
        print(f'Epoch [{epoch}]/[{num_epochs}]')
        checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)

        model.eval()
        translate = translate_sentence(model, sentence, chinese, english, device, max_length=50)
        print(f'Translated example sentence \n {translate}')
        model.train()

        loop = tqdm(enumerate(train_iterator),total=len(train_iterator),leave=False)
        epoch_loss = 0
        for batch_idx, batch in loop:
            inp_data = batch.ch.to(device)
            target = batch.eng.to(device)
            output = model(inp_data, target)
            #output shape: (trg_len, batch_size, output_dim)
            output = output[1:].reshape(-1,output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            epoch_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1) #梯度裁剪
            optimizer.step()
            writer.add_scalar('batch Loss', loss, global_step=step)
            step += 1
            loop.set_description(f'Epoch[{epoch}/{num_epochs}]')
        writer.add_scalar('epoch Loss', epoch_loss, epoch)
        print("Total loss: {}".format(epoch_loss))

if __name__=="__main__":
    train()