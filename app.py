import streamlit as st
import numpy as np
import os
from music21 import stream, note

# 1. Force Ultra-Wide Desktop Layout
st.set_page_config(
    page_title="AI Music Composer Dashboard",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. High-Contrast Theme & Interactive CSS
st.markdown("""
    <style>
    /* Main body background styling */
    .stApp {
        background: linear-gradient(135deg, #09070f 0%, #0f0b1e 50%, #050308 100%);
        color: #f0f0f5;
    }
    
    /* Center container titles */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(45deg, #00f2fe, #4facfe, #9b51e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #a5b4fc;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 30px;
    }
    
    /* High-Contrast Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* Enhanced Contrast Headings inside Cards */
    .card-heading {
        color: #00f2fe !important;
        font-weight: 700 !important;
        margin-bottom: 15px !important;
        letter-spacing: 0.5px;
    }
    
    /* Crisp White Text for Sliders and Labels */
    .stSlider label, .stMarkdown p, label {
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    
    /* Animated Note Badges */
    .note-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b1d75, #4facfe);
        color: #ffffff;
        padding: 8px 14px;
        margin: 5px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 15px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 4px 12px rgba(0, 242, 254, 0.15);
    }
    
    /* Virtual Piano Styles with Deep Glow */
    .piano-container {
        display: flex;
        justify-content: center;
        background: #09070f;
        padding: 25px;
        border-radius: 14px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 0 30px #000;
        margin-top: 15px;
    }
    .piano-key {
        width: 44px;
        height: 160px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #111;
        border-radius: 0 0 6px 6px;
        margin: 2px;
        transition: all 0.2s ease;
    }
    .piano-key.active {
        background: linear-gradient(to bottom, #00f2fe, #9b51e0);
        box-shadow: 0 0 20px #00f2fe, inset 0 -10px 20px rgba(255,255,255,0.4);
        transform: translateY(2px);
    }
    
    /* Custom CSS for Streamlit Base Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #09070f !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.5) !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("<h1 class='main-title'>🎵 AI MUSIC COMPOSER DESKTOP PANEL</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Recurrent Pattern Matrix Engine for Generative Audio Engineering</p>", unsafe_allow_html=True)

# Dataset Initialization
TRAINING_NOTES = [
    "C4", "E4", "G4", "C5", "G4", "E4", "C4",
    "F4", "A4", "C5", "F5", "C5", "A4", "F4",
    "G4", "B4", "D5", "G5", "D5", "B4", "G4",
    "C4", "E4", "G4", "C5", "E5", "G5", "C6"
]
pitches = sorted(list(set(TRAINING_NOTES)))
note_to_int = {note: number for number, note in enumerate(pitches)}
int_to_note = {number: note for number, note in enumerate(pitches)}
n_vocab = len(pitches)

# Compile probability matrix
transition_matrix = np.zeros((n_vocab, n_vocab))
for i in range(len(TRAINING_NOTES) - 1):
    transition_matrix[note_to_int[TRAINING_NOTES[i]], note_to_int[TRAINING_NOTES[i+1]]] += 1
for i in range(n_vocab):
    row_sum = np.sum(transition_matrix[i])
    transition_matrix[i] = transition_matrix[i] / row_sum if row_sum > 0 else np.ones(n_vocab) / n_vocab

# Two-Column Layout Setup
left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    # Configurations Card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='card-heading'>⚙️ Engine Configurations</h3>", unsafe_allow_html=True)
    
    creativity = st.slider("🧠 Softmax Temperature (Creativity Index)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    generation_length = st.slider("🎼 Composition Length (Total Notes)", min_value=10, max_value=40, value=25, step=5)
    st.markdown("</div>", unsafe_allow_html=True)
    
    trigger_generation = st.button("✨ Initialize Generative Audio Run", use_container_width=True)

with right_col:
    if trigger_generation:
        # Results Card - Generated dynamically so no empty shapes show up beforehand
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-heading'>🔮 Live Generation Stream</h3>", unsafe_allow_html=True)
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        for percent in range(100):
            progress_bar.progress(percent + 1)
            status_text.text(f"🧬 Processing probabilistic matrices... {percent + 1}%")
            
        current_note_idx = np.random.randint(0, n_vocab)
        prediction_output = [int_to_note[current_note_idx]]
        
        for _ in range(generation_length - 1):
            probs = transition_matrix[current_note_idx]
            probs = np.log(probs + 1e-8) / creativity
            exp_probs = np.exp(probs)
            scaled_probs = exp_probs / np.sum(exp_probs)
            current_note_idx = np.random.choice(n_vocab, p=scaled_probs)
            prediction_output.append(int_to_note[current_note_idx])
            
        status_text.empty()
        progress_bar.empty()
        st.success("🤖 Composition generation sequence successful!")
        
        # Audio Synthesizer Engine
        st.markdown("<h4 style='color: #ffffff; margin-top: 15px;'>🔊 Live Audio Synthesizer</h4>", unsafe_allow_html=True)
        sr = 44100
        duration_per_note = 0.25
        t = np.linspace(0, duration_per_note, int(sr * duration_per_note), endpoint=False)
        note_freqs = {"C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99, "C6": 1046.50}
        
        audio_wave = []
        for p in prediction_output:
            freq = note_freqs.get(p, 440.0)
            wave = np.sin(2 * np.pi * freq * t)
            audio_wave.extend(wave)
            
        audio_signal = np.array(audio_wave)
        st.audio(audio_signal, sample_rate=sr)
        
        # Predicted Sequence Stream Output
        st.markdown("<h4 style='color: #ffffff; margin-top: 15px;'>🎼 Predicted Sequence Stream:</h4>", unsafe_allow_html=True)
        badges_html = "".join([f"<span class='note-badge'>{n}</span>" for n in prediction_output])
        st.markdown(badges_html, unsafe_allow_html=True)
        
        # Virtual Interactive Piano Layer
        st.markdown("<h4 style='color: #ffffff; margin-top: 15px;'>🎹 Matrix Active Keys (Visual Synthesizer Output)</h4>", unsafe_allow_html=True)
        active_unique_notes = set(prediction_output)
        
        piano_html = "<div class='piano-container'>"
        piano_scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5", "C6"]
        for k in piano_scale:
            is_active = "active" if k in active_unique_notes else ""
            piano_html += f"<div class='piano-key {is_active}' title='{k}'></div>"
        piano_html += "</div>"
        st.markdown(piano_html, unsafe_allow_html=True)
        
        # Compilation and Export File Processing
        output_notes = []
        for p in prediction_output:
            new_note = note.Note(p)
            new_note.quarterLength = 0.5
            output_notes.append(new_note)
            
        midi_stream = stream.Stream(output_notes)
        output_filename = "generated_ai_music.mid"
        midi_stream.write('midi', fp=output_filename)
        
        if os.path.exists(output_filename):
            with open(output_filename, "rb") as file:
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Export Generated Track to MIDI File",
                    data=file,
                    file_name="ai_music_track.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # High-Contrast Welcome Card (Replaces the unneeded blank containers)
        st.markdown("<div class='glass-card' style='text-align: center; padding: 75px 30px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #00f2fe; margin-bottom: 10px;'>🛰️ System Awaiting Execution</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a5b4fc; font-size: 15px;'>Adjust composition lengths or variation vectors on the left panel, then trigger the engine initialization run to display live visual analytics and synthesized streams.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)