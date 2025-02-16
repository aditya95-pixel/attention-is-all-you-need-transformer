import streamlit as st
import torch
from model import build_transformer
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from dataset import causal_mask

def load_model(config, device):
    tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file("tokenizer_it.json")
    
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config['seq_len'], 
        config['seq_len'], 
        d_model=config['d_model']
    ).to(device)
    
    model_path = get_weights_file_path(config, "19")  # Load latest checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    
    return model, tokenizer_src, tokenizer_tgt

def translate_text(model, tokenizer_src, tokenizer_tgt, text, device, max_len=50):
    src_tokens = tokenizer_src.encode(text).ids
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    src_mask = torch.ones_like(src_tensor).to(device)
    
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    encoder_output = model.encode(src_tensor, src_mask)
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        next_word = torch.argmax(prob, dim=1).item()
        
        if next_word == eos_idx:
            break
        
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], dtype=torch.long).to(device)], dim=1)
    
    translated_text = tokenizer_tgt.decode(decoder_input.squeeze(0).tolist())
    return translated_text

def main():
    st.title("English to Italian Translator")
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer_src, tokenizer_tgt = load_model(config, device)
    
    text_input = st.text_area("Enter text in English:")
    if st.button("Translate"):
        if text_input.strip():
            translated_text = translate_text(model, tokenizer_src, tokenizer_tgt, text_input, device)
            st.write("**Translation:**", translated_text)
        else:
            st.warning("Please enter text to translate.")

if __name__ == "__main__":
    main()
