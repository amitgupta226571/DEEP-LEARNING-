import torch
import torch.nn as nn
import numpy as np
from preprocess import preprocess_sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(real, pred, criterion):
    # PyTorch CrossEntropyLoss expects (N, C) where C is classes
    loss = criterion(pred, real)
    mask = real.ge(1).type(torch.float32) # ignore padding index 0
    loss_ = loss * mask
    return torch.mean(loss_)

def train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang_tokenizer, optimizer, criterion, batch_size):
    loss = 0
    optimizer.zero_grad()

    enc_output, dec_hidden = encoder(inp, enc_hidden)

    # Decoder input shape: (batch_size, 1)
    dec_input = torch.tensor([[targ_lang_tokenizer.word2idx['<start>']]] * batch_size, device=device)

    # Teacher forcing
    for t in range(1, targ.size(1)):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions, criterion)
        # Teacher forcing step
        dec_input = targ[:, t].unsqueeze(1)

    batch_loss = (loss.item() / int(targ.size(1)))
    loss.backward()
    optimizer.step()

    return batch_loss

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    encoder.eval()
    decoder.eval()

    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx.get(i, 0) for i in sentence.split(' ')]
    inputs = torch.tensor(inputs, device=device).unsqueeze(0) # add batch dimension

    result = ''
    with torch.no_grad():
        hidden = (torch.zeros(1, 1, encoder.enc_units, device=device),
                  torch.zeros(1, 1, encoder.enc_units, device=device))

        enc_out, dec_hidden = encoder(inputs, hidden)
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]], device=device)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

            if attention_weights is not None:
                attention_weights = attention_weights.view(-1).cpu().numpy()
                attention_plot[t, :len(attention_weights)] = attention_weights

            predicted_id = predictions.argmax(1).item()
            result += targ_lang.idx2word[predicted_id] + ' '

            if targ_lang.idx2word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            dec_input = torch.tensor([[predicted_id]], device=device)

    return result, sentence, attention_plot
