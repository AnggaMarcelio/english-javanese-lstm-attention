import streamlit as st
import tensorflow as tf
import numpy as np
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import time
from datetime import datetime

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from keras.layers import Layer

# =====================================================================================
# 1. DEFINISI CUSTOM LAYER (ATTENTION) - LOGIKA TETAP
# =====================================================================================

class AttentionLayer(Layer):
    """Custom Attention Layer (Bahdanau)"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name="att_weight_encoder", shape=(input_shape[0][2], input_shape[0][2]), initializer="glorot_uniform", trainable=False)
        self.W2 = self.add_weight(name="att_weight_decoder", shape=(input_shape[1][2], input_shape[0][2]), initializer="glorot_uniform", trainable=False)
        self.V = self.add_weight(name="att_weight_combine", shape=(input_shape[0][2], 1), initializer="glorot_uniform", trainable=False)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        encoder_outputs, decoder_outputs = inputs
        decoder_expanded = tf.expand_dims(decoder_outputs, 2)
        encoder_expanded = tf.expand_dims(encoder_outputs, 1)

        score = tf.nn.tanh(tf.matmul(encoder_expanded, self.W1) + tf.matmul(decoder_expanded, self.W2))
        score = tf.matmul(score, self.V)

        if mask is not None and mask[0] is not None:
            encoder_mask = tf.cast(mask[0], dtype=tf.float32)
            mask_expanded = tf.reshape(encoder_mask, (tf.shape(encoder_mask)[0], 1, tf.shape(encoder_mask)[1], 1))
            score = score + (1.0 - mask_expanded) * -1e9

        attention_weights = tf.nn.softmax(score, axis=2)
        context = attention_weights * encoder_expanded
        context_vector = tf.reduce_sum(context, axis=2)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        return context_vector, attention_weights

    def get_config(self):
        return super(AttentionLayer, self).get_config()

def preprocess_english(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"([?.!,¿'])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z0-9?.!,¿'\s-]", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =====================================================================================
# 2. FUNGSI MEMUAT MODEL, TOKENIZER, DAN DATASET
# =====================================================================================
@st.cache_resource
def load_dataset():
    """Memuat dataset translasi dari file .en dan .jv."""
    try:
        with open('dataset/dataset-english.en', 'r', encoding='utf-8') as f_en, \
             open('dataset/dataset-javanese.jv', 'r', encoding='utf-8') as f_jv:
            
            eng_sentences = f_en.readlines()
            jv_sentences = f_jv.readlines()
            
            if len(eng_sentences) != len(jv_sentences):
                st.error("Jumlah baris di file dataset tidak cocok!")
                return None

            # Preprocess English sentences and create a lookup dictionary
            dataset_lookup = {
                preprocess_english(eng.strip()): jv.strip() 
                for eng, jv in zip(eng_sentences, jv_sentences)
            }
            return dataset_lookup
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan. Pastikan folder 'dataset' ada.")
        return None

@st.cache_resource
def load_translation_models():
    """Memuat encoder, decoder, dan tokenizer."""
    custom_objects = {'AttentionLayer': AttentionLayer}

    # Load Models
    encoder_model = keras.models.load_model('encoder_inference_model.keras', 
                                            custom_objects=custom_objects,
                                            compile=False)
    
    decoder_model = keras.models.load_model('decoder_inference_model.keras', 
                                            custom_objects=custom_objects,
                                            compile=False)

    # Load Tokenizers
    with open('eng_tokenizer.json') as f:
        eng_tokenizer = tokenizer_from_json(f.read())
        
    with open('jv_tokenizer.json') as f:
        jv_tokenizer = tokenizer_from_json(f.read())
        
    # Fix integer keys
    new_index_word = {}
    for k, v in jv_tokenizer.index_word.items():
        try:
            new_index_word[int(k)] = v
        except ValueError:
            new_index_word[k] = v
    jv_tokenizer.index_word = new_index_word

    with open('model_config.json') as f:
        config = json.load(f)

    return encoder_model, decoder_model, eng_tokenizer, jv_tokenizer, config

# =====================================================================================
# 3. FUNGSI PENERJEMAHAN DAN EVALUASI
# =====================================================================================

def translate_sentence(input_sentence, encoder_model, decoder_model, eng_tokenizer, jv_tokenizer, config):
    """Menerjemahkan kalimat dan mengembalikan informasi debug."""
    max_len_eng = config['max_len_eng']
    max_len_jv = config['max_len_jv']

    def flatten_to_numpy(data):
        flat = []
        if isinstance(data, (list, tuple)):
            for item in data:
                flat.extend(flatten_to_numpy(item))
        else:
            if hasattr(data, 'numpy'):
                flat.append(data.numpy())
            else:
                flat.append(data)
        return flat

    # 1. Preprocess
    processed_input = preprocess_english(input_sentence)
    input_words = processed_input.split()
    
    # 2. Tokenize Input
    input_seq = eng_tokenizer.texts_to_sequences([processed_input])
    input_padded = pad_sequences(input_seq, maxlen=max_len_eng, padding='post')

    # 3. Encoder Prediction
    enc_raw_output = encoder_model.predict(input_padded, verbose=0)
    
    # 4. Extract Encoder Outputs & States
    flat_enc_outputs = flatten_to_numpy(enc_raw_output)
    
    encoder_outputs = None
    state_h = None
    state_c = None
    states_2d = []

    for item in flat_enc_outputs:
        if not hasattr(item, 'shape'): continue
        if len(item.shape) == 3:
             if item.shape[1] > 1:
                encoder_outputs = item
        elif len(item.shape) == 2:
            states_2d.append(item)

    if len(states_2d) >= 2:
        state_h, state_c = states_2d[0], states_2d[1]
    elif len(states_2d) == 1:
        state_h, state_c = states_2d[0], states_2d[0]
    
    if encoder_outputs is None:
        return {'translation': "Error: Encoder output failed.", 'attention': None, 'input_tokens': []}

    # 5. Persiapan Decoder
    target_seq = np.zeros((1, 1), dtype=np.float32)
    start_token = jv_tokenizer.word_index.get('<start>', 1) 
    target_seq[0, 0] = start_token
    
    decoded_tokens = []
    attention_weights_list = []
    stop_condition = False
    iteration = 0

    # 6. Loop Decoding
    while not stop_condition and iteration < max_len_jv:
        iteration += 1
        decoder_inputs_predict = [target_seq, state_h, state_c, encoder_outputs]
        raw_decoder_outputs = decoder_model.predict(decoder_inputs_predict, verbose=0)
        flat_outputs = flatten_to_numpy(raw_decoder_outputs)

        output_tokens, new_states, attention = None, [], None

        for out in flat_outputs:
            if not hasattr(out, 'shape'): continue
            if len(out.shape) == 3:
                if out.shape[-1] > 500: 
                    output_tokens = out
                elif out.shape[1] == 1: 
                    attention = out
            elif len(out.shape) == 2:
                new_states.append(out)

        if output_tokens is None: break

        if attention is not None:
            attention_weights_list.append(attention[0, 0, :])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = jv_tokenizer.index_word.get(int(sampled_token_index), '')
        
        if sampled_word == '<end>' or sampled_word == '':
            stop_condition = True
        else:
            decoded_tokens.append(sampled_word)
        
        target_seq[0, 0] = sampled_token_index
        
        if len(new_states) >= 2:
            state_h, state_c = new_states[0], new_states[1]
        
    # 7. Susun Hasil
    translation_text = ' '.join(decoded_tokens)
    attention_matrix = np.array(attention_weights_list) if attention_weights_list and decoded_tokens else None
    
    return {
        'translation': translation_text,
        'attention': attention_matrix,
        'input_tokens': input_words
    }

# =====================================================================================
# 4. UI STREAMLIT - CLEAN & INTERACTIVE
# =====================================================================================

st.set_page_config(
    layout="wide", 
    page_title="NMT: English to Javanese",
    page_icon="✨",
    initial_sidebar_state="expanded" # Kunci agar sidebar terbuka saat awal
)

# -------------------------------------------------------------------------------------
# UPDATED CSS: FIXED SIDEBAR, NO ARROWS, MODERN LOOK
# -------------------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Import Google Fonts - Plus Jakarta Sans for a modern feel */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    /* --- SIDEBAR CONFIGURATION (FIXED & CLEAN) --- */
    
    /* 1. Hide the Arrow / Collapse Button */
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    /* Hide the drag handle container at the top which often contains the X */
    section[data-testid="stSidebar"] > div > div:first-child {{
        visibility: hidden;
        height: 0px;
    }}

    /* 2. Modern Sidebar Container Styling */
    section[data-testid="stSidebar"] {{
        background-color: #F8FAFC; /* Light gray background */
        border-right: 1px solid #E2E8F0; /* Subtle border */
        box-shadow: 2px 0 12px rgba(0,0,0,0.02); /* Very soft shadow */
    }}

    /* 3. Padding and Spacing inside Sidebar */
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 2rem !important;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }}
    
    /* 4. Sidebar Heading Style */
    [data-testid="stSidebar"] h2 {{
        color: #1E293B;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* --- MAIN AREA STYLING --- */
    
    .main .block-container {{
        padding: 2rem 3rem;
        max-width: 1400px;
    }}
    
    /* Hide Streamlit Default Branding */
    #MainMenu, footer {{
        visibility: hidden;
    }}

    /* --- COMPONENT STYLING --- */
    
    /* Text Area Styling */
    .stTextArea textarea {{
        border-radius: 10px;
        border: 1px solid #CBD5E0;
        background-color: #FFFFFF;
        font-size: 16px;
        transition: all 0.2s;
    }}
    .stTextArea textarea:focus {{
        border-color: #4A90E2;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
    }}
    
    /* Button Styling */
    .stButton>button {{
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: transform 0.1s;
    }}
    .stButton>button:active {{
        transform: scale(0.98);
    }}
    .stButton>button[kind="primary"] {{
        background-color: #4A90E2;
        color: white;
        box-shadow: 0 4px 6px rgba(74, 144, 226, 0.2);
    }}
    .stButton>button[kind="secondary"] {{
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }}
    .stButton>button[kind="secondary"]:hover {{
        background-color: #E2E8F0;
    }}

    /* Card/Divider Aesthetics */
    hr {{
        margin: 2rem 0;
        border-color: #E2E8F0;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 8px;
        background-color: transparent;
        color: #64748B;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #EFF6FF;
        color: #4A90E2;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================================================
# SESSION STATE INITIALIZATION
# =====================================================================================
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = {}
if 'input_text_key' not in st.session_state:
    st.session_state.input_text_key = ""


# =====================================================================================
# SIDEBAR
# =====================================================================================

with st.sidebar:
    # JUDUL BARU: Lebih profesional & spesifik
    st.markdown("<h2 style='text-align: center;'>🧠 English-Javanese Neural Machine Translation</h2>", unsafe_allow_html=True)
    
    # DESKRIPSI BARU: Menjelaskan model (LSTM + Attention)
    st.markdown("""
    <div style='text-align: justify; color: #475569; font-size: 0.9em; line-height: 1.5; margin-bottom: 20px; padding: 0 5px;'>
        Sistem penerjemahan otomatis yang dibangun menggunakan arsitektur 
        <b>Sequence-to-Sequence (Seq2Seq)</b> dengan <b>LSTM</b> dan 
        <b>Attention Mechanism</b>.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Quick Examples")
    examples = [
        "Don't smoke.",
        "I'm buying souvenirs for my family.",
        "She goes to the market",
        "Where do you live?",
        "I'm very hungry."
    ]
    
    for i, text in enumerate(examples):
        if st.button(text, key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text_key = text
            st.rerun()
            
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    with st.container(border=True):
        show_heatmap = st.toggle("Show Heatmap", value=True, help="Visualisasi bobot atensi model saat menerjemahkan.")

# =====================================================================================
# MAIN CONTENT
# =====================================================================================

# Load Model & Dataset
try:
    with st.spinner("Loading translation model & dataset..."):
        encoder, decoder, eng_tokenizer, jv_tokenizer, config = load_translation_models()
        dataset_lookup = load_dataset()
    model_loaded = True if encoder and dataset_lookup else False
except Exception as e:
    st.error(f"❌ Failed to load model or dataset: {e}")
    model_loaded = False

# Header
st.markdown("<h1 style='text-align: center; font-weight: 700; color: #1E293B;'>Machine Translation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em; color: #64748B;'>Penerjemahan Bahasa Inggris ke Bahasa Jawa menggunakan Sequence-to-Sequence LSTM dengan Attention</p>", unsafe_allow_html=True)

st.divider()

# Main Translation Interface
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Input Bahasa Inggris")
    input_text = st.text_area(
        "Enter English text to translate",
        height=200, 
        key="input_text_key",
        placeholder="Ketik atau tempel teks di sini...\nContoh: I want to sleep early tonight",
        label_visibility="collapsed"
    )
    
    translate_btn = st.button(
        "🚀 Terjemahkan", 
        type="primary", 
        use_container_width=True, 
        disabled=not model_loaded
    )

with col2:
    st.markdown("### Hasil Terjemahan (Jawa)")
    
    translation_output = st.session_state.translation_result.get('translation', '')

    st.text_area(
        "Hasil terjemahan akan muncul di sini...",
        value=translation_output,
        height=200,
        disabled=False,
        label_visibility="collapsed"
    )

# Translation Logic
if translate_btn and model_loaded:
    if not input_text.strip():
        st.warning("Input tidak boleh kosong!")
    else:
        with st.spinner("Translating... (Encoder → Attention → Decoder)"):
            try:
                start_time = time.time()
                result_dict = translate_sentence(input_text, encoder, decoder, eng_tokenizer, jv_tokenizer, config)
                end_time = time.time()
                
                translation_time = end_time - start_time
                st.session_state.translation_result = result_dict
                
                history_item = {
                    'input': input_text,
                    'output': result_dict.get('translation', 'Error'),
                    'time': translation_time,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.translation_history.insert(0, history_item)
                
                st.toast(f"✅ Finished in {translation_time:.2f}s!", icon="🎉")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred during translation: {e}")
                st.session_state.translation_result = {}

st.divider()

# =====================================================================================
# ANALYSIS, BATCH PROCESSING & HISTORY TABS
# =====================================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Analysis", "📝 Token Details", "🕒 History", "📊 Evaluasi Model"])

with tab1:
    if not st.session_state.translation_result.get('translation'):
        st.info("Terjemahkan kalimat untuk melihat Analisisnya.")
    elif not show_heatmap:
        st.info("Heatmap is disabled. Enable it in the sidebar settings to view.")
    else:
        att_mat = st.session_state.translation_result.get('attention')
        in_tok = st.session_state.translation_result.get('input_tokens', [])
        out_tok = st.session_state.translation_result.get('translation', '').split()

        if att_mat is not None and in_tok and out_tok:
            try:
                fig, ax = plt.subplots(figsize=(max(8, len(in_tok) * 0.6), max(6, len(out_tok) * 0.4)))
                
                sns.heatmap(
                    att_mat[:len(out_tok), :len(in_tok)],
                    xticklabels=in_tok, yticklabels=out_tok,
                    cmap="viridis", linewidths=0.5, ax=ax, annot=True, fmt=".2f"
                )
                plt.xticks(rotation=45, ha="right")
                plt.title("Heatmap Visualization", fontsize=14, pad=20)
                plt.xlabel("Input Tokens (English)")
                plt.ylabel("Output Tokens (Javanese)")
                st.pyplot(fig)
                st.caption("Darker colors indicate higher attention weights from the input to generate the output.")
            except Exception as e:
                st.warning(f"⚠️ Failed to render heatmap: {e}")
        else:
            st.warning("⚠️ Not enough attention data available for visualization.")

with tab2:
    if not st.session_state.translation_result.get('translation'):
        st.info("Terjemahkan kalimat untuk melihat Detail Token.")
    else:
        st.markdown("### 📝 Token Information")
        
        tok_col1, tok_col2 = st.columns(2)
        in_tok = st.session_state.translation_result.get('input_tokens', [])
        out_tok = st.session_state.translation_result.get('translation', '').split()

        with tok_col1:
            st.metric("Input Tokens", len(in_tok))
            st.json(in_tok)
        with tok_col2:
            st.metric("Output Tokens", len(out_tok))
            st.json(out_tok)

with tab3:
    st.markdown("### 🕒 Translation History")
    if not st.session_state.translation_history:
        st.info("Riwayat terjemahan Kosong.")
    else:
        for i, item in enumerate(st.session_state.translation_history):
            with st.container():
                st.markdown(f"<small>*{item['timestamp']}*</small>", unsafe_allow_html=True)
                hist_col1, hist_col2 = st.columns(2)
                with hist_col1:
                    st.text_area("Input", value=item['input'], key=f"hist_in_{item['timestamp']}", disabled=True, height=100)
                with hist_col2:
                    st.text_area("Output", value=item['output'], key=f"hist_out_{item['timestamp']}", disabled=True, height=100)
                
                st.divider()

with tab4:
    st.markdown("### 📊 Evaluasi Model & Batch Testing")
    st.markdown("""
    Fitur ini digunakan untuk mengukur performa model penerjemah menggunakan metrik **BLEU Score (N-Gram 1-4)**.
    1.  **Mode Evaluasi:** Upload file **.CSV** yang berisi kolom `english` (input) dan `javanese` (referensi).
    2.  **Mode Prediksi:** Upload file **.TXT** berisi kalimat Bahasa Inggris saja.
    """)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Dataset (.csv) atau Text (.txt)", 
        type=['csv', 'txt'], 
        label_visibility="collapsed"
    )

    if uploaded_file:
        df_input = None
        sentences = []
        ground_truths = [] 
        
        # --- A. PROSES PEMBACAAN DATA ---
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
                # Normalisasi nama kolom
                df_input.columns = [c.lower() for c in df_input.columns]
                
                if 'english' in df_input.columns and 'javanese' in df_input.columns:
                    sentences = df_input['english'].astype(str).tolist()
                    ground_truths = df_input['javanese'].astype(str).tolist()
                    st.success(f"📂 Dataset CSV dimuat: {len(sentences)} data.")
                else:
                    st.error("⚠️ Format CSV tidak valid. Pastikan kolom 'english' dan 'javanese' tersedia.")
            
            elif uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                sentences = [s.strip() for s in content.split('\n') if s.strip()]
                ground_truths = [None] * len(sentences) 
                st.info(f"📄 File Teks dimuat: {len(sentences)} kalimat.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

        # --- B. TOMBOL EKSEKUSI ---
        if sentences:
            if st.button("🚀 Mulai Evaluasi", type="primary"):
                prog = st.progress(0, text="Memproses data...")
                
                # Penampung skor per kategori
                scores = {'b1': [], 'b2': [], 'b3': [], 'b4': []}
                results = []
                
                start_time_batch = time.time()
                smoothie = SmoothingFunction().method7

                for i, input_text in enumerate(sentences):
                    prog.progress((i + 1) / len(sentences), text=f"Processing {i+1}/{len(sentences)}...")
                    
                    # 1. Proses Translasi (Prediksi)
                    try:
                        trans_res = translate_sentence(input_text, encoder, decoder, eng_tokenizer, jv_tokenizer, config)
                        prediction = trans_res['translation']
                    except:
                        prediction = "Error"

                    # 2. Logika Evaluasi
                    reference = "-"
                    # Inisialisasi skor 0
                    b_scores = [0.0, 0.0, 0.0, 0.0] 

                    # Cek Referensi (dari CSV atau Lookup Internal)
                    current_ref = ground_truths[i]
                    if current_ref is None and dataset_lookup:
                        key = preprocess_english(input_text)
                        current_ref = dataset_lookup.get(key)
                    
                    # Jika Referensi Ditemukan -> Hitung BLEU
                    if current_ref:
                        reference = current_ref
                        
                        ref_tokens = [reference.split()]
                        pred_tokens = prediction.split()

                        # Hitung BLEU 1-4
                        weights_list = [
                            (1.0, 0.0, 0.0, 0.0),          # BLEU-1
                            (0.5, 0.5, 0.0, 0.0),      # BLEU-2
                            (0.333, 0.333, 0.333, 0.0), # BLEU-3
                            (0.25, 0.25, 0.25, 0.25) # BLEU-4
                        ]
                        
                        for idx, w in enumerate(weights_list):
                            # Gunakan try-except untuk menangani error perhitungan
                            try:
                                score = sentence_bleu(ref_tokens, pred_tokens, weights=w, smoothing_function=smoothie)
                            except:
                                score = 0.0
                            
                            # Pastikan skor tidak pernah lebih dari 1.0
                            score = min(score, 1.0) 
                            
                            b_scores[idx] = score

                        scores['b1'].append(b_scores[0])
                        scores['b2'].append(b_scores[1])
                        scores['b3'].append(b_scores[2])
                        scores['b4'].append(b_scores[3])

                    results.append({
                        "Input": input_text,
                        "Prediction": prediction,
                        "Ground Truth": reference,
                        "BLEU-1": b_scores[0],
                        "BLEU-2": b_scores[1],
                        "BLEU-3": b_scores[2],
                        "BLEU-4": b_scores[3]
                    })
                
                duration = time.time() - start_time_batch
                prog.progress(1.0, text="Selesai!")

                # --- C. TAMPILAN SCORE CARD (BLEU 1-4) ---
                if scores['b4']: 
                    st.success(f"✅ Evaluasi selesai dalam {duration:.2f} detik.")

                    # CSS untuk Score Card yang rapi
                    st.markdown("""
                    <style>
                    .metric-container {
                        background-color: white; border: 1px solid #e2e8f0; 
                        border-radius: 8px; padding: 15px; text-align: center;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 10px;
                    }
                    .metric-label { font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
                    .metric-value { font-size: 1.8rem; color: #1e293b; font-weight: 700; margin-top: 5px; }
                    </style>
                    """, unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    metrics_info = [
                        ("BLEU-1", np.mean(scores['b1'])),
                        ("BLEU-2", np.mean(scores['b2'])),
                        ("BLEU-3", np.mean(scores['b3'])),
                        ("BLEU-4", np.mean(scores['b4']))
                    ]
                    
                    cols = [c1, c2, c3, c4]
                    for col, (label, val) in zip(cols, metrics_info):
                        with col:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{val:.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak ada data referensi (Ground Truth) yang ditemukan untuk menghitung skor.")

                # --- D. TABEL DETAIL (Polos Tanpa Highlight) ---
                st.markdown("### 📋 Detail Hasil")
                
                df_res = pd.DataFrame(results)

                # Render Tabel Polos (Hanya format angka)
                st.dataframe(
                    df_res.style.format({
                        "BLEU-1": "{:.4f}", 
                        "BLEU-2": "{:.4f}", 
                        "BLEU-3": "{:.4f}", 
                        "BLEU-4": "{:.4f}"
                    }),
                    use_container_width=True
                )

                # Download Button
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Laporan (.csv)",
                    data=csv,
                    file_name=f"laporan_evaluasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

st.markdown("<div style='margin-top: 5rem;'></div>", unsafe_allow_html=True)