import gradio as gr
import json
import tempfile
import os
from turktoken import TurkishBPETokenizer

# Global tokenizer instance
tokenizer = TurkishBPETokenizer()
is_trained = False

def train_tokenizer(text, vocab_size):
    global tokenizer, is_trained
    if not text or len(text.strip()) < 10:
        return "Please enter a text with at least 10 characters.", "", ""
    
    try:
        tokenizer = TurkishBPETokenizer()
        tokenizer.train(text, vocab_size=int(vocab_size))
        is_trained = True
        
        vocab_count = len(tokenizer.vocab)
        merge_count = len(tokenizer.merges)
        
        sample_tokens = list(tokenizer.vocab.items())[:20]
        sample_display = "\n".join([f"ID {k}: {repr(v.decode('utf-8', errors='replace'))}" for k, v in sample_tokens if k >= 256])
        
        stats = f"""Tokenizer trained successfully!

Statistics:
• Vocabulary size: {vocab_count}
• Merge count: {merge_count}
• Base byte tokens: 256
• Learned tokens: {vocab_count - 256}
"""
        return stats, sample_display if sample_display else "No tokens learned yet", ""
    except Exception as e:
        return f"Training error: {str(e)}", "", ""

def add_special_tokens(tokens_text):
    global tokenizer, is_trained
    if not is_trained:
        return "Train the tokenizer first!"
    
    try:
        tokens = [t.strip() for t in tokens_text.split(",") if t.strip()]
        if not tokens:
            return "Please enter at least one special token (comma separated)"
        
        tokenizer.add_special_tokens(tokens)
        
        special_list = "\n".join([f"• {k}: ID {v}" for k, v in tokenizer.special_tokens.items()])
        return f"{len(tokens)} special tokens added!\n\nSpecial Tokens:\n{special_list}"
    except Exception as e:
        return f"Error: {str(e)}"

def encode_text(text):
    global tokenizer, is_trained
    if not is_trained:
        return "Train the tokenizer first!", ""
    
    if not text:
        return "Please enter some text.", ""
    
    try:
        ids = tokenizer.encode(text)
        
        token_details = []
        for idx in ids:
            if idx in tokenizer.vocab:
                token_bytes = tokenizer.vocab[idx]
                token_str = token_bytes.decode('utf-8', errors='replace')
                token_details.append(f"[{idx}] -> {repr(token_str)}")
        
        details = "\n".join(token_details)
        return f"Token IDs ({len(ids)} tokens):\n{ids}", f"Token Details:\n{details}"
    except Exception as e:
        return f"Encoding error: {str(e)}", ""

def decode_ids(ids_text):
    global tokenizer, is_trained
    if not is_trained:
        return "Train the tokenizer first!"
    
    try:
        ids_text = ids_text.strip()
        if ids_text.startswith("[") and ids_text.endswith("]"):
            ids = json.loads(ids_text)
        else:
            ids = [int(x.strip()) for x in ids_text.replace(",", " ").split() if x.strip()]
        
        decoded = tokenizer.decode(ids)
        return f"Decoded text:\n{decoded}"
    except Exception as e:
        return f"Decoding error: {str(e)}"

def export_tokenizer():
    global tokenizer, is_trained
    if not is_trained:
        return None, "Train the tokenizer first!"
    
    try:
        export_data = {
            "merges": {f"{p0} {p1}": idx for (p0, p1), idx in tokenizer.merges.items()},
            "special_tokens": tokenizer.special_tokens,
            "vocab_size": len(tokenizer.vocab),
            "info": {
                "name": "TurkToken",
                "version": "0.1.0",
                "description": "Turkish-optimized BPE tokenizer"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            temp_path = f.name
        
        json_preview = json.dumps(export_data, ensure_ascii=False, indent=2)[:2000]
        if len(json.dumps(export_data)) > 2000:
            json_preview += "\n... (truncated)"
        
        return temp_path, f"Tokenizer exported successfully!\n\nJSON Preview:\n```json\n{json_preview}\n```"
    except Exception as e:
        return None, f"Export error: {str(e)}"

def get_vocab_display():
    global tokenizer, is_trained
    if not is_trained:
        return "Train the tokenizer first!"
    
    try:
        vocab_items = []
        for idx, token_bytes in sorted(tokenizer.vocab.items()):
            if idx >= 256:
                token_str = token_bytes.decode('utf-8', errors='replace')
                vocab_items.append(f"ID {idx}: {repr(token_str)}")
        
        if not vocab_items:
            return "No learned tokens yet."
        
        return "\n".join(vocab_items[:100]) + (f"\n\n... and {len(vocab_items) - 100} more tokens" if len(vocab_items) > 100 else "")
    except Exception as e:
        return f"Error: {str(e)}"


# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em !important;
    font-weight: bold;
    margin-bottom: 0.5em;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2em;
}
.stat-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 15px;
}
footer {visibility: hidden}
"""

# Sample Turkish text
sample_text = """Türkiye, Avrupa ve Asya kıtalarının kesişim noktasında yer alan eşsiz bir ülkedir. 
İstanbul, Boğaziçi'nin iki yakasını birbirine bağlayan köprüleriyle dünyaca ünlüdür.
Türk mutfağı, kebaplardan baklava ve künefe gibi tatlılara kadar zengin çeşitliliğiyle bilinir.

Atatürk, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır.
Ankara, Türkiye'nin başkenti ve ikinci büyük şehridir.
Kapadokya'nın peri bacaları, her yıl milyonlarca turisti ağırlamaktadır.

Türkçe, Ural-Altay dil ailesine mensup, zengin bir dildir.
Özellikle ğ, ş, ç, ı, ö, ü gibi karakterler Türkçe'ye özgüdür.
Bu tokenizer, Türkçe metinleri verimli şekilde işlemek için tasarlanmıştır.
"""

# Build the interface
with gr.Blocks(head=f"<style>{custom_css}</style>", title="TurkToken - Turkish BPE Tokenizer") as demo:
    gr.HTML("""
        <h1 class="main-header">TurkToken</h1>
        <p class="sub-header">Turkish-optimized Byte Pair Encoding (BPE) Tokenizer</p>
    """)
    
    with gr.Tabs():
        # TAB 1: Training
        with gr.TabItem("Training", id=1):
            gr.Markdown("### Train tokenizer with your own text")
            
            with gr.Row():
                with gr.Column(scale=2):
                    train_text = gr.Textbox(
                        label="Training Text",
                        placeholder="Enter Turkish text or use the sample...",
                        lines=10,
                        value=sample_text
                    )
                    with gr.Row():
                        vocab_size = gr.Slider(
                            minimum=300,
                            maximum=10000,
                            value=1000,
                            step=100,
                            label="Vocabulary Size"
                        )
                        train_btn = gr.Button("Start Training", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    train_output = gr.Textbox(label="Training Result", lines=8, interactive=False)
                    learned_tokens = gr.Textbox(label="Learned Tokens (First 20)", lines=6, interactive=False)
            
            train_btn.click(
                train_tokenizer,
                inputs=[train_text, vocab_size],
                outputs=[train_output, learned_tokens, gr.Textbox(visible=False)]
            )
        
        # TAB 2: Encode/Decode
        with gr.TabItem("Encode / Decode", id=2):
            gr.Markdown("### Convert text to tokens or tokens to text")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Encode")
                    encode_input = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to encode...",
                        lines=3,
                        value="Merhaba dünya! Bu bir test cümlesidir."
                    )
                    encode_btn = gr.Button("Encode", variant="primary")
                    encode_output = gr.Textbox(label="Token IDs", lines=3, interactive=False)
                    token_details = gr.Textbox(label="Token Details", lines=6, interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### Decode")
                    decode_input = gr.Textbox(
                        label="Token IDs",
                        placeholder="Enter content like [256, 312] or 256 312...",
                        lines=3
                    )
                    decode_btn = gr.Button("Decode", variant="primary")
                    decode_output = gr.Textbox(label="Decoded Text", lines=5, interactive=False)
            
            encode_btn.click(encode_text, inputs=[encode_input], outputs=[encode_output, token_details])
            decode_btn.click(decode_ids, inputs=[decode_input], outputs=[decode_output])
        
        # TAB 3: Special Tokens
        with gr.TabItem("Special Tokens", id=3):
            gr.Markdown("### Add special tokens (BOS, EOS, PAD etc.)")
            
            with gr.Row():
                with gr.Column():
                    special_input = gr.Textbox(
                        label="Special Tokens (comma separated)",
                        placeholder="<|bos|>, <|eos|>, <|pad|>, <|unk|>",
                        value="<|bos|>, <|eos|>, <|pad|>, <|unk|>, <|sep|>"
                    )
                    add_special_btn = gr.Button("Add Special Tokens", variant="primary")
                
                with gr.Column():
                    special_output = gr.Textbox(label="Result", lines=8, interactive=False)
            
            add_special_btn.click(add_special_tokens, inputs=[special_input], outputs=[special_output])
        
        # TAB 4: Vocabulary
        with gr.TabItem("Vocabulary", id=4):
            gr.Markdown("### View all learned tokens")
            
            vocab_display = gr.Textbox(label="Vocabulary (ID 256+)", lines=20, interactive=False)
            refresh_vocab_btn = gr.Button("Refresh Vocabulary", variant="secondary")
            
            refresh_vocab_btn.click(get_vocab_display, outputs=[vocab_display])
        
        # TAB 5: Export
        with gr.TabItem("Export", id=5):
            gr.Markdown("### Download trained tokenizer as JSON")
            
            with gr.Row():
                with gr.Column():
                    export_btn = gr.Button("Export Tokenizer", variant="primary", size="lg")
                    export_file = gr.File(label="Download File")
                
                with gr.Column():
                    export_preview = gr.Markdown(label="JSON Preview")
            
            export_btn.click(export_tokenizer, outputs=[export_file, export_preview])
    
    gr.Markdown("""
    ---
    ### User Guide
    
    1. **Training**: Enter your Turkish text and set vocabulary size
    2. **Encode/Decode**: Convert text <-> tokens with the trained tokenizer
    3. **Special Tokens**: Add special tokens like BOS, EOS
    4. **Vocabulary**: View all learned tokens
    5. **Export**: Download the tokenizer configuration
    
    ---
    <p style="text-align: center; color: #888;">
        Made by <a href="https://github.com/hsperus">hsperus</a> | 
        <a href="https://pypi.org/project/turktoken/">PyPI</a> | 
        <a href="https://github.com/hsperus/turktoken">GitHub</a>
    </p>
    """)

if __name__ == "__main__":
    demo.launch()
