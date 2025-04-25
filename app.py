import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import ClapProcessor, ClapModel
import parselmouth
from parselmouth.praat import call
import pywt
import io
import os
import asyncio
from PIL import Image
import streamlit.components.v1 as components
from scipy import spatial
import base64

st.set_page_config(
    page_title="Voice Stress Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css_and_js():
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #f0f2f6;
    }
    h1, h2, h4 {
        color: #2c3e50;
    }
    .main-container {
        padding: 40px 60px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stFileUploader>div>div {
        border: 2px dashed #4CAF50;
        border-radius: 8px;
        padding: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f5e9;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 25px 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.3s;
        margin-bottom: 20px;
    }
    .feature-card:hover {
        transform: scale(1.02);
    }
    .feature-icon {
        font-size: 32px;
        margin-bottom: 12px;
    }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        console.log("App UI loaded");
    });
    </script>
    """, unsafe_allow_html=True)

def show_feature_cards():
    st.markdown("""
    <div style='margin-bottom: 30px;'>
        <h3>🚀 Key Features </h3>
    </div>
""", unsafe_allow_html=True)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🔊</div>
            <h4>CLAP Audio Embeddings</h4>
            <p>Transformer-based embeddings that capture audio patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🧠</div>
            <h4>Handcrafted Features</h4>
            <p>Includes MFCC, Jitter, Spectrogram, Spectral Flex, and more.</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🧪</div>
            <h4>Dual & Single Audio Analysis</h4>
            <p>Compare two audios or analyze one for stress detection.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom: 30px;'>
    </div>
""", unsafe_allow_html=True)

MODEL_NAME = "clap_model"
MAX_DURATION = 6.0
SAMPLE_RATE = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

@st.cache_resource
def load_models():
    try:
        processor = ClapProcessor.from_pretrained(MODEL_NAME)
        clap_model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE)
        
        class StressClassifier(torch.nn.Module):
            def __init__(self, clap_model):
                super().__init__()
                self.clap = clap_model
                self.audio_emb_size = clap_model.config.projection_dim
                self.text_emb_size = clap_model.config.projection_dim

                self.classifier = torch.nn.Sequential(
                    torch.nn.LazyLinear(1024),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(1024, 512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(512, 256),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(256, 1)
                )

            def forward(self, input_features, input_ids, attention_mask, stress_features):
                audio_emb = self.clap.get_audio_features(input_features=input_features)
                text_emb = self.clap.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                combined = torch.cat([audio_emb, text_emb, stress_features], dim=1)
                return self.classifier(combined)

        model = StressClassifier(clap_model).to(DEVICE)
        model.load_state_dict(torch.load("best_stress_model.pth", map_location=DEVICE))
        model.eval()
        
        return processor, model
        
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

processor, model = load_models()

def extract_f0(audio, sr):
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=500)
        f0 = f0[~np.isnan(f0)]
        return {
            'f0_mean': np.mean(f0) if len(f0) > 0 else 0,
            'f0_std': np.std(f0) if len(f0) > 0 else 0,
            'f0_range': np.ptp(f0) if len(f0) > 0 else 0
        }
    except:
        return {'f0_mean': 0, 'f0_std': 0, 'f0_range': 0}

def extract_formants(audio, sr):
    try:
        sound = parselmouth.Sound(audio, sr)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 50, 500)
        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        num_points = call(pointProcess, "Get number of points")

        f1, f2, f3, f4 = [], [], [], []
        for point in range(1, num_points + 1):
            t = call(pointProcess, "Get time from index", point)
            f1.append(call(formants, "Get value at time", 1, t, 'HERTZ', 'LINEAR'))
            f2.append(call(formants, "Get value at time", 2, t, 'HERTZ', 'LINEAR'))
            f3.append(call(formants, "Get value at time", 3, t, 'HERTZ', 'LINEAR'))
            f4.append(call(formants, "Get value at time", 4, t, 'HERTZ', 'LINEAR'))

        return {
            'f1_mean': np.nanmean(f1) if f1 else 0,
            'f1_std': np.nanstd(f1) if f1 else 0,
            'f2_mean': np.nanmean(f2) if f2 else 0,
            'f2_std': np.nanstd(f2) if f2 else 0,
            'f3_mean': np.nanmean(f3) if f3 else 0,
            'f3_std': np.nanstd(f3) if f3 else 0,
            'f4_mean': np.nanmean(f4) if f4 else 0,
            'f4_std': np.nanstd(f4) if f4 else 0
        }
    except:
        return {'f1_mean': 0, 'f1_std': 0, 'f2_mean': 0, 'f2_std': 0,
                'f3_mean': 0, 'f3_std': 0, 'f4_mean': 0, 'f4_std': 0}

def extract_chirp_features(audio, sr):
    try:
        coeffs = pywt.wavedec(audio, 'cmor1.5-1.0', level=5)
        cA = coeffs[0]
        return {'chirp_mean': np.mean(cA), 'chirp_std': np.std(cA), 'chirp_energy': np.sum(cA**2)}
    except:
        return {'chirp_mean': 0, 'chirp_std': 0, 'chirp_energy': 0}

def extract_fft_spectrogram(audio, sr):
    try:
        spectrogram = np.abs(librosa.stft(audio))
        fft_features = []
        for i in range(spectrogram.shape[0]):
            fft_features.extend([np.mean(spectrogram[i,:]), np.std(spectrogram[i,:]), np.max(spectrogram[i,:])])
        return fft_features[:30]
    except:
        return [0]*30

def extract_mfcc(audio, sr):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_features = []
        for i in range(mfcc.shape[0]):
            mfcc_features.extend([np.mean(mfcc[i,:]), np.std(mfcc[i,:]), np.max(mfcc[i,:])])
        return mfcc_features[:39]
    except:
        return [0]*39

def extract_jitter_shimmer(audio, sr):
    try:
        sound = parselmouth.Sound(audio, sr)
        point_process = call(sound, "To PointProcess (periodic, cc)", 50, 500)
        return {
            'jitter_local': call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'jitter_rap': call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            'jitter_ppq5': call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'shimmer_local': call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'shimmer_apq3': call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'shimmer_apq5': call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        }
    except:
        return {'jitter_local': 0, 'jitter_rap': 0, 'jitter_ppq5': 0,
                'shimmer_local': 0, 'shimmer_apq3': 0, 'shimmer_apq5': 0}

def extract_hnr(audio, sr):
    try:
        sound = parselmouth.Sound(audio, sr)
        hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        return {'hnr': call(hnr, "Get mean", 0, 0)}
    except:
        return {'hnr': 0}

def extract_spectral_flux(audio, sr):
    try:
        spectrogram = np.abs(librosa.stft(audio))
        flux = librosa.onset.onset_strength(S=spectrogram)
        return {'spectral_flux_mean': np.mean(flux), 'spectral_flux_std': np.std(flux), 'spectral_flux_max': np.max(flux)}
    except:
        return {'spectral_flux_mean': 0, 'spectral_flux_std': 0, 'spectral_flux_max': 0}

def extract_spectral_centroid(audio, sr):
    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return {'spectral_centroid_mean': np.mean(centroid), 'spectral_centroid_std': np.std(centroid), 'spectral_centroid_max': np.max(centroid)}
    except:
        return {'spectral_centroid_mean': 0, 'spectral_centroid_std': 0, 'spectral_centroid_max': 0}

def preprocess_audio(audio, sr):
    try:
        # Pad audio
        if len(audio) < MAX_DURATION * SAMPLE_RATE:
            audio = np.pad(audio, (0, int(MAX_DURATION * SAMPLE_RATE) - len(audio)))
        else:
            audio = audio[:int(MAX_DURATION * SAMPLE_RATE)]

        # Extract all features
        f0 = extract_f0(audio, sr)
        formants = extract_formants(audio, sr)
        chirp = extract_chirp_features(audio, sr)
        fft_spectrogram = extract_fft_spectrogram(audio, sr)
        mfcc = extract_mfcc(audio, sr)
        jitter_shimmer = extract_jitter_shimmer(audio, sr)
        hnr = extract_hnr(audio, sr)
        spectral_flux = extract_spectral_flux(audio, sr)
        spectral_centroid = extract_spectral_centroid(audio, sr)
        rmse = librosa.feature.rms(y=audio)[0]

        # Combine all features
        features = [
            f0['f0_mean'], f0['f0_std'], f0['f0_range'],
            formants['f1_mean'], formants['f1_std'],
            formants['f2_mean'], formants['f2_std'],
            formants['f3_mean'], formants['f3_std'],
            formants['f4_mean'], formants['f4_std'],
            chirp['chirp_mean'], chirp['chirp_std'], chirp['chirp_energy'],
            *fft_spectrogram,
            *mfcc,
            jitter_shimmer['jitter_local'], jitter_shimmer['jitter_rap'], jitter_shimmer['jitter_ppq5'],
            jitter_shimmer['shimmer_local'], jitter_shimmer['shimmer_apq3'], jitter_shimmer['shimmer_apq5'],
            hnr['hnr'],
            spectral_flux['spectral_flux_mean'], spectral_flux['spectral_flux_std'], spectral_flux['spectral_flux_max'],
            spectral_centroid['spectral_centroid_mean'], spectral_centroid['spectral_centroid_std'],
            spectral_centroid['spectral_centroid_max'],
            np.mean(rmse)
        ]

        audio_input = processor(
            audios=audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).to(DEVICE)

        text_input = processor(
            text="This is a speech sample",
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        return {
            'input_features': audio_input['input_features'],
            'input_ids': text_input['input_ids'],
            'attention_mask': text_input['attention_mask'],
            'stress_features': torch.tensor([features], dtype=torch.float32).to(DEVICE),
            'audio': audio,
            'sr': sr,
            'features': features
        }
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def predict_stress(inputs):
    # Make prediction
    with torch.no_grad():
        output = model(
            input_features=inputs['input_features'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            stress_features=inputs['stress_features']
        )
        prob = torch.sigmoid(output).item()
        prediction = "Stressed" if prob > 0.5 else "Not Stressed"

    return {
        "prediction": prediction,
        "probability": prob,
        "confidence": prob if prediction == "Stressed" else 1 - prob
    }

def plot_waveform(audio, sr, title="Waveform", color='#1f77b4'):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, color=color)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    st.pyplot(plt)

def plot_mel_spectrogram(audio, sr, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 3))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

def plot_fft_spectrogram(audio, sr, title="FFT Spectrogram"):
    plt.figure(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='plasma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

def plot_chirp_spectrogram(audio, sr, title="Chirplet Transform Spectrogram"):
    plt.figure(figsize=(10, 3))
    coeffs, freqs = pywt.cwt(audio, np.arange(1, 128), 'cmor1.5-1.0')
    plt.imshow(np.abs(coeffs), extent=[0, len(audio)/sr, 1, 128], 
               cmap='magma', aspect='auto', vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
    plt.colorbar()
    plt.title(title)
    plt.ylabel('Scale')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    st.pyplot(plt)




def plot_spectral_features(audio, sr, title="Spectral Features"):
    # Spectral Centroid and Bandwidth
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # Compute the time variable for visualization
    frames = range(len(spectral_centroid))
    t = librosa.frames_to_time(frames, sr=sr)
    
    plt.figure(figsize=(10, 4))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.4, color='#1f77b4')
    plt.plot(t, spectral_centroid, color='r', label='Spectral Centroid')
    plt.plot(t, spectral_bandwidth, color='g', label='Spectral Bandwidth')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.ylabel('Hz')
    
    # Plot RMSE
    plt.subplot(2, 1, 2)
    rmse = librosa.feature.rms(y=audio)[0]
    plt.plot(t, rmse, color='b', label='RMSE')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    st.pyplot(plt)

def plot_mfcc(audio, sr, title="MFCCs"):
    plt.figure(figsize=(10, 3))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

def compare_features(features1, features2, feature_names):
    similarity_scores = {}
    for name, feat1, feat2 in zip(feature_names, features1, features2):
        if isinstance(feat1, (list, np.ndarray)) and isinstance(feat2, (list, np.ndarray)):
            # For array-like features, calculate cosine similarity
            similarity = 1 - spatial.distance.cosine(feat1, feat2)
        else:
            # For scalar features, calculate normalized difference
            max_val = max(abs(feat1), abs(feat2))
            if max_val == 0:
                similarity = 1.0
            else:
                similarity = 1 - (abs(feat1 - feat2) / max_val)
        similarity_scores[name] = max(0, min(1, similarity))  # Ensure between 0 and 1
    
    return similarity_scores

def display_comparison_results(audio1, audio2, sr1, sr2, inputs1, inputs2):
    st.markdown("## 🆚 Audio Comparison Results")
    
    # Feature comparison
    feature_names = [
        'F0 Mean', 'F0 Std', 'F0 Range',
        'F1 Mean', 'F1 Std', 'F2 Mean', 'F2 Std',
        'F3 Mean', 'F3 Std', 'F4 Mean', 'F4 Std',
        'Chirp Mean', 'Chirp Std', 'Chirp Energy',
        *[f'FFT Band {i}' for i in range(1, 31)],
        *[f'MFCC {i}' for i in range(1, 40)],
        'Jitter Local', 'Jitter RAP', 'Jitter PPQ5',
        'Shimmer Local', 'Shimmer APQ3', 'Shimmer APQ5',
        'HNR', 'Spectral Flux Mean', 'Spectral Flux Std', 'Spectral Flux Max',
        'Spectral Centroid Mean', 'Spectral Centroid Std', 'Spectral Centroid Max',
        'RMSE'
    ]
    
    similarity_scores = compare_features(inputs1['features'], inputs2['features'], feature_names)
    avg_similarity = np.mean(list(similarity_scores.values()))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Audio 1")
        st.audio(audio1, format='audio/wav', sample_rate=sr1)
    with col2:
        st.markdown(f"### Audio 2")
        st.audio(audio2, format='audio/wav', sample_rate=sr2)
    
    st.markdown(f"### Overall Similarity: {avg_similarity:.2%}")
    st.progress(avg_similarity)
    
    # Visual comparison tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Waveform", "Mel Spectrogram", "FFT Spectrogram", 
        "Chirplet Transform", "Spectral Features", "MFCC"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            plot_waveform(audio1, sr1, "Audio 1 Waveform", '#1f77b4')
        with col2:
            plot_waveform(audio2, sr2, "Audio 2 Waveform", '#ff7f0e')
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            plot_mel_spectrogram(audio1, sr1, "Audio 1 Mel Spectrogram")
        with col2:
            plot_mel_spectrogram(audio2, sr2, "Audio 2 Mel Spectrogram")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            plot_fft_spectrogram(audio1, sr1, "Audio 1 FFT Spectrogram")
        with col2:
            plot_fft_spectrogram(audio2, sr2, "Audio 2 FFT Spectrogram")
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            plot_chirp_spectrogram(audio1, sr1, "Audio 1 Chirplet Transform")
        with col2:
            plot_chirp_spectrogram(audio2, sr2, "Audio 2 Chirplet Transform")
    
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            plot_spectral_features(audio1, sr1, "Audio 1 Spectral Features")
        with col2:
            plot_spectral_features(audio2, sr2, "Audio 2 Spectral Features")
    
    with tab6:
        col1, col2 = st.columns(2)
        with col1:
            plot_mfcc(audio1, sr1, "Audio 1 MFCCs")
        with col2:
            plot_mfcc(audio2, sr2, "Audio 2 MFCCs")
    
    # Key feature differences
    st.markdown("### 🔍 Key Feature Differences")
    
    # Get top 5 most different features
    sorted_features = sorted(similarity_scores.items(), key=lambda x: x[1])
    most_different = sorted_features[:5]
    most_similar = sorted_features[-5:]
    
    col1, col2 = st.columns(2)
    col1, col2 = st.columns(2)
    import random

    icons_different = ["🚫", "⚠️", "📉", "🔺", "🛑", "❗", "🥵", "🔧", "😓", "📛"]
    icons_similar = ["✅", "📈", "👍", "🟢", "🌱", "🔒", "💡", "🎯", "🔁", "🧩"]

    # Card and hover style
    card_style = """
        <div style="background-color: #ffffff; padding: 5px; border-radius: 15px; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin-bottom: 20px; 
                    transition: all 0.3s ease; cursor: pointer;">
        <h3 style="font-size: 22px; font-weight: bold; text-align: center; margin-bottom: 15px;">
    """

    hover_style = """
        <style>
            .hover-card:hover {
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
                transform: translateY(-5px);
            }
        </style>
    """

    with col1:
        st.markdown(hover_style, unsafe_allow_html=True)
        st.markdown(f'<div class="hover-card">{card_style}<span style="color: #d32f2f;">🔻 Most Different Features</span></h3>', unsafe_allow_html=True)

        for i, feature in enumerate(most_different):
            feature_name = feature[0]  # Extract feature name from the tuple
            icon = icons_different[i % len(icons_different)]
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center; padding-bottom: 5px; margin: 5px 0;">
                    <span style="font-size: 28px; margin-bottom: 5px;">{icon}</span>
                    <strong style="color: #000000; font-size: 18px; text-align: center;">{feature_name}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(hover_style, unsafe_allow_html=True)
        st.markdown(f'<div class="hover-card">{card_style}<span style="color: #388e3c;">🟢 Most Similar Features</span></h3>', unsafe_allow_html=True)

        for i, feature in enumerate(most_similar):
            feature_name = feature[0]  # Extract feature name from the tuple
            icon = icons_similar[i % len(icons_similar)]
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center; padding-bottom: 5px; margin: 5px 0;">
                    <span style="font-size: 28px; margin-bottom: 5px;">{icon}</span>
                    <strong style="color: #000000; font-size: 18px; text-align: center;">{feature_name}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div></div>", unsafe_allow_html=True)



def display_results(result, audio, sr):
    
    st.markdown("### 🎙️ Analysis Result")

        

    if result['prediction'] == "Stressed":
        st.markdown("""
            <div style='
                background-color: #ffdddd;
                padding: 10px;
                border-radius: 10px;
                border: 1px solid #ff4b4b;
                max-width: 80%;
                margin-right: auto;
                margin-top: 15px;
                margin-bottom:10px;
            '>
                <h2 style='color: #ff4b4b; text-align: center;'>⚠️ STRESS DETECTED</h2>
                <p style='text-align: center;'>The speaker appears to be under stress (angry, fearful, or sad).</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
                <div style='
                    background-color: #ddffdd;
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #4CAF50;
                    max-width: 80%;
                    margin-right: auto;
                    margin-top: 15px;
                    margin-bottom:10px;
                '>
                    <h2 style='color: #4CAF50; text-align: center;'>✅ NO STRESS DETECTED</h2>
                    <p style='text-align: center;'>The speaker appears calm and unstressed.</p>
                </div>
                """, unsafe_allow_html=True)


    # Confidence Bar
    st.markdown("""
        <div style='margin-top: 30px;'>
            <h4>📊 Confidence Level</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Confidence:** {result['confidence']:.2%}")
    st.progress(result['confidence'])

    # Gauge Canvas for Probability
    gauge_html = f"""
    <canvas id='gaugeCanvas' width='200' height='150'></canvas>
    <script>
    const canvas = document.getElementById('gaugeCanvas');
    const ctx = canvas.getContext('2d');
    const value = {result['probability']};

    function drawGauge(value) {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.arc(100, 100, 80, 0.75 * Math.PI, 2.25 * Math.PI);
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 20;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(100, 100, 80, 0.75 * Math.PI, 0.75 * Math.PI + (1.5 * Math.PI * value));
        ctx.strokeStyle = value > 0.5 ? '#ff4b4b' : '#4CAF50';
        ctx.lineWidth = 20;
        ctx.stroke();

        ctx.fillStyle = '#333';
        ctx.font = '20px Arial';
        ctx.fillText((value * 100).toFixed(1) + '%', 80, 110);
    }}
    drawGauge(value);
    </script>
    """
    components.html(gauge_html, height=200)

    # Probability bar (horizontal)
    st.markdown("#### 📈 Stress Probability")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(0, result['probability'], color='#ff4b4b' if result['prediction'] == "Stressed" else '#4CAF50')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0%', '50%', '100%'])
    ax.set_yticks([])
    ax.set_title('Stress Probability')
    st.pyplot(fig)

def audio_to_base64(audio, sr):
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    audio_bytes = buffer.getvalue()
    return base64.b64encode(audio_bytes).decode()

# Inject JS and Confetti
def inject_custom_js():
    custom_js = """
    <script>
    function confetti() {
        const duration = 2000;
        const end = Date.now() + duration;

        (function frame() {
            const confettiSettings = {
                particleCount: 7,
                spread: 55,
                origin: { y: 0.6 }
            };
            window.confetti(Object.assign({}, confettiSettings, { angle: 60, origin: { x: 0 } }));
            window.confetti(Object.assign({}, confettiSettings, { angle: 120, origin: { x: 1 } }));

            if (Date.now() < end) requestAnimationFrame(frame);
        }());
    }

    document.addEventListener('DOMContentLoaded', function () {
        console.log("Custom JS Loaded");
    });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    """
    components.html(custom_js, height=0)

def display_results2(result, audio, sr):
    # Main result display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        inject_custom_js()

        st.markdown("### 🎙️ Analysis Result")

        

        if result['prediction'] == "Stressed":
            st.markdown("""
                <div style='
                    background-color: #ffdddd;
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #ff4b4b;
                    max-width: 80%;
                    margin-right: auto;
                    margin-top: 15px;
                    margin-bottom:10px;
                '>
                    <h2 style='color: #ff4b4b; text-align: center;'>⚠️ STRESS DETECTED</h2>
                    <p style='text-align: center;'>The speaker appears to be under stress (angry, fearful, or sad).</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
                    <div style='
                        background-color: #ddffdd;
                        padding: 10px;
                        border-radius: 10px;
                        border: 1px solid #4CAF50;
                        max-width: 80%;
                        margin-right: auto;
                        margin-top: 15px;
                        margin-bottom:10px;
                    '>
                        <h2 style='color: #4CAF50; text-align: center;'>✅ NO STRESS DETECTED</h2>
                        <p style='text-align: center;'>The speaker appears calm and unstressed.</p>
                    </div>
                    """, unsafe_allow_html=True)


        # Confidence Bar
        st.markdown("""
            <div style='margin-top: 30px;'>
                <h4>📊 Confidence Level</h4>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
        st.progress(result['confidence'])

        # Gauge Canvas for Probability
        gauge_html = f"""
        <canvas id='gaugeCanvas' width='200' height='150'></canvas>
        <script>
        const canvas = document.getElementById('gaugeCanvas');
        const ctx = canvas.getContext('2d');
        const value = {result['probability']};

        function drawGauge(value) {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            ctx.arc(100, 100, 80, 0.75 * Math.PI, 2.25 * Math.PI);
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 20;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(100, 100, 80, 0.75 * Math.PI, 0.75 * Math.PI + (1.5 * Math.PI * value));
            ctx.strokeStyle = value > 0.5 ? '#ff4b4b' : '#4CAF50';
            ctx.lineWidth = 20;
            ctx.stroke();

            ctx.fillStyle = '#333';
            ctx.font = '20px Arial';
            ctx.fillText((value * 100).toFixed(1) + '%', 80, 110);
        }}
        drawGauge(value);
        </script>
        """
        components.html(gauge_html, height=200)

        # Probability bar (horizontal)
        st.markdown("#### 📈 Stress Probability")
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(0, result['probability'], color='#ff4b4b' if result['prediction'] == "Stressed" else '#4CAF50')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0%', '50%', '100%'])
        ax.set_yticks([])
        ax.set_title('Stress Probability')
        st.pyplot(fig)

    
        with col2:
            st.markdown("""
                <div style='margin-top: 40px; margin-bottom: 30px;'>
                    <h3>📊 Audio Visualizations</h3>
                </div>
                """, unsafe_allow_html=True)

            tab1, tab2, tab3, tab4,tab5,tab6 = st.tabs(["Waveform", "Mel Spectrogram","FFT Spectrogram","Chirp Transform","Spectral Features", "MFCC"])
            
            with tab1:
                plot_waveform(audio, sr)
            
            with tab2:
                plot_mel_spectrogram(audio, sr)
            with tab3:
                plot_fft_spectrogram(audio, sr)
            with tab4:
                plot_chirp_spectrogram(audio, sr)
            
            with tab5:
                plot_spectral_features(audio, sr)
            
            with tab6:
                plot_mfcc(audio, sr)

def single_audio_analysis():
    st.markdown("## 🔍 Single Audio Analysis")
    
    
    audio = None
    sr = None
    
    uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3"], key="single_upload")
    
    if uploaded_file is not None:
        try:
            audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
            st.success("Audio file loaded successfully!")
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
    
    
    if audio is not None:
        # Show audio info
        duration = len(audio) / sr
        st.info(f"Audio Info: {duration:.2f} seconds, {sr} Hz sample rate")
        
        # Process and analyze button
        if st.button("Analyze Stress Level", key="single_analyze", use_container_width=True):
            with st.spinner("Processing audio and analyzing..."):
                try:
                    # Preprocess audio
                    inputs = preprocess_audio(audio, sr)
                    
                    if inputs is not None:
                        # Make prediction
                        result = predict_stress(inputs)
                        
                        # Display results
                        display_results2(result, inputs['audio'], inputs['sr'])
                        
                        
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

def compare_audio_analysis():
    st.markdown("## 🆚 Audio Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Audio File 1")
        audio_file1 = st.file_uploader("Choose first audio file", type=["wav", "mp3"], key="compare_upload1")
    
    with col2:
        st.markdown("### Audio File 2")
        
        audio_file2 = st.file_uploader("Choose second audio file", type=["wav", "mp3"], key="compare_upload2")
    
    if audio_file1 is not None and audio_file2 is not None:
        try:
            # Load both audio files
            audio1, sr1 = librosa.load(audio_file1, sr=SAMPLE_RATE)
            audio2, sr2 = librosa.load(audio_file2, sr=SAMPLE_RATE)
            
            # Show audio info
            col1, col2 = st.columns(2)
            with col1:
                duration1 = len(audio1) / sr1
                st.info(f"Audio 1 Info: {duration1:.2f}s, {sr1}Hz")
            with col2:
                duration2 = len(audio2) / sr2
                st.info(f"Audio 2 Info: {duration2:.2f}s, {sr2}Hz")
            
            if st.button("Compare Audio Files", use_container_width=True):
                with st.spinner("Processing and comparing audio files..."):
                    try:
                        # Preprocess both audio files
                        inputs1 = preprocess_audio(audio1, sr1)
                        inputs2 = preprocess_audio(audio2, sr2)
                        
                        if inputs1 is not None and inputs2 is not None:
                            # Display comparison results
                            display_comparison_results(
                                inputs1['audio'], inputs2['audio'], 
                                inputs1['sr'], inputs2['sr'],
                                inputs1, inputs2
                            )
                            
                            # Also show individual stress analysis
                            st.markdown("## ⚡Individual Stress Analysis")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### Audio 1 Stress Analysis")
                                result1 = predict_stress(inputs1)
                                display_results(result1, inputs1['audio'], inputs1['sr'])
                            
                            with col2:
                                st.markdown("### Audio 2 Stress Analysis")
                                result2 = predict_stress(inputs2)
                                display_results(result2, inputs2['audio'], inputs2['sr'])
                            
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading audio files: {str(e)}")

def main():

    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 25px;">
        <h1 style="margin: 0;">Voice Stress Detection</h1>
        <span style="font-size: 30px; margin-left: 10px;">🎤</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<pre>
<div style="background-color: #f9f9f9; padding: 25px; border-radius: 15px; border: 1px solid #e0e0e0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 40px;">

    <h3 style="color: #333333; margin-bottom: 20px;">🧠 How It Works</h3>

    <p style="font-size: 18px; line-height: 1.6; margin-bottom: 20px;">
        This application detects <strong style="color: #4CAF50;">Stress in speech</strong> using a 
        <strong style="color: #2196F3;">Fusion-Based Model</strong> that combines:
    </p>

    <ul style="font-size: 17px; line-height: 1.8; padding-left: 20px;">
        <ul>🔊 <strong>CLAP Audio Embeddings</strong> — Transformer-based representations of sound</ul>
        <ul>💬 <strong>Text Embeddings</strong> extracted from transcriptions</ul>
        <ul>🧬 <strong>Handcrafted Features</strong>: MFCC, Formants, Spectrogram, FFT, Chirp Spectra, Spectral Flex, Spectral Centroid, HNR, Jitter, Shimmer</ul>
    </ul>

    <p style="font-size: 18px; line-height: 1.6; margin-top: 30px;">
    You can either analyze a 
    <strong style="color: #4CAF50;">Single audio</strong> or 
    <strong style="color: #2196F3;">compare Two audios</strong> 
    to identify possible stress indicators in speech.
</p>


</div>
</pre>
""", unsafe_allow_html=True)




    show_feature_cards()
    local_css_and_js()

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'sr' not in st.session_state:
        st.session_state.sr = None

    tab1, tab2 = st.tabs(["🎧 Single Audio Analysis", "🎙️ Dual Audio Analysis"])

    with tab1:
        single_audio_analysis()

    with tab2:
        compare_audio_analysis()

    st.markdown("</div>", unsafe_allow_html=True)  

if __name__ == "__main__":
    main()
