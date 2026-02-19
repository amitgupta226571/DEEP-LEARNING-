import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

# Import modules
from preprocess import load_dataset, tokenize
from models import Encoder, Decoder, device
from engine import train_step, evaluate

st.set_page_config(page_title="NeuTrans: PyTorch Seq2Seq", page_icon="🇪🇸", layout="wide")

st.markdown("""
<style>
    .stApp { background: #301934; }
    h1 { color: #d32f2f; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; text-align: center; }
    .stButton>button { background-color: #d32f2f; color: white; border-radius: 20px; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# Configuration
st.sidebar.header("⚙️ Configuration")
num_examples = st.sidebar.slider("Dataset Size", 1000, 30000, 5000)
EPOCHS = st.sidebar.number_input("Training Epochs", min_value=1, max_value=20, value=5)
BATCH_SIZE = 64
embedding_dim = 256
units = 512

model_mode = st.sidebar.selectbox("Model Architecture", ["LSTM + Attention", "LSTM (No Attention)"])
use_attn = True if model_mode == "LSTM + Attention" else False

DATA_PATH = 'data/spa.txt'
targ_lang, inp_lang = load_dataset(DATA_PATH, num_examples)

if targ_lang is None:
    st.error(f"File {DATA_PATH} not found.")
    st.stop()

input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

max_length_inp, max_length_targ = input_tensor.shape[1], target_tensor.shape[1]

input_train, _, target_train, _ = train_test_split(input_tensor, target_tensor, test_size=0.2)

# PyTorch DataLoader
train_data = TensorDataset(torch.tensor(input_train, dtype=torch.long), torch.tensor(target_train, dtype=torch.long))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

vocab_inp_size = inp_lang_tokenizer.vocab_size
vocab_tar_size = targ_lang_tokenizer.vocab_size

if 'trained' not in st.session_state:
    st.session_state.trained = False

st.title("🇪🇸 English-to-Spanish NMT (PyTorch)")

if st.sidebar.button("🚀 Train Model"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE).to(device)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, use_attention=use_attn).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    criterion = nn.CrossEntropyLoss(reduction='none')

    encoder.train()
    decoder.train()

    steps_per_epoch = len(train_loader)

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, (inp, targ) in enumerate(train_loader):
            inp, targ = inp.to(device), targ.to(device)
            enc_hidden = encoder.initialize_hidden_state()

            batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang_tokenizer, optimizer, criterion, BATCH_SIZE)
            total_loss += batch_loss

        status_text.text(f'Epoch {epoch + 1} | Loss: {total_loss / steps_per_epoch:.4f}')
        progress_bar.progress((epoch + 1) / EPOCHS)

    st.session_state.encoder = encoder
    st.session_state.decoder = decoder
    st.session_state.trained = True
    st.success("Training Complete!")

col_input, col_output = st.columns(2)

with col_input:
    input_text = st.text_area("Input (English)", "How are you?")

with col_output:
    if st.button("Translate"):
        if not st.session_state.trained:
            st.warning("⚠️ Train the model first!")
        else:
            result, sentence, attention_plot = evaluate(
                input_text, st.session_state.encoder, st.session_state.decoder,
                inp_lang_tokenizer, targ_lang_tokenizer, max_length_inp, max_length_targ
            )
            st.success(result)

            if use_attn:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 1, 1)
                ax.matshow(attention_plot[:len(result.split(' ')), :len(sentence.split(' '))], cmap='viridis')
                ax.set_xticklabels([''] + sentence.split(' '), rotation=90)
                ax.set_yticklabels([''] + result.split(' '))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                st.pyplot(fig)
