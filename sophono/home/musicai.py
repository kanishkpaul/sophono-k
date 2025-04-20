import os
import numpy as np
import librosa
import librosa.display
import tkinter as tk
from tkinter import filedialog
import json
import matplotlib.pyplot as plt
from groq import Groq
import re
import math

def select_mp3_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an MP3 file",
        filetypes=[("MP3 files", "*.mp3")]
    )
    return file_path

def extract_acoustic_features(mp3_path):
    y, sr = librosa.load(mp3_path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(v) for v in np.mean(mfccs, axis=1)]

    features = {
        "tempo": float(tempo),
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zero_crossing_rate,
        "rms": rms,
    }
    
    print("\nMFCC Features (Mean Values):")
    mfcc_names = [
        "1. Brightness (High Frequency Energy)",
        "2. Sharpness (Attack Quality)",
        "3. Instrument Harshness",
        "4. Vocal Clarity",
        "5. Mid-Range Punch",
        "6. Bass Presence",
        "7. Warmth (Low-Mid Balance)",
        "8. Metallic/Organic Balance",
        "9. Harmonic Complexity",
        "10. Fundamental Bass",
        "11. Room Acoustics",
        "12. Ultra-Low Rumble",
        "13. Spectral Shape"
    ]
    
    for i, (name, val) in enumerate(zip(mfcc_names, mfcc_mean), start=1):
        print(f"{name}: {val:.4f}")
        features[f"mfcc_{i}"] = val
        
    return features

def analyze_viral_trend(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    
    tempo = float(librosa.beat.tempo(y=audio, sr=sr)[0])
    beat_frames = librosa.beat.beat_track(y=audio, sr=sr)[1]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    rms = librosa.feature.rms(y=audio)[0]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    segments = librosa.segment.agglomerative(mfccs, k=10)
    unique_segments = len(np.unique(segments))
    avg_rms = float(np.mean(rms))
    
    score = 0
    tempo_check = 110 <= tempo <= 130
    if tempo_check: score += 1
    
    energy_check = avg_rms > 0.1
    if energy_check: score += 1
    
    repetition_check = unique_segments < 8
    if repetition_check: score += 1
    
    highlight_duration = 15
    try:
        energy_per_second = rms[:len(audio)//sr]
        if len(energy_per_second) >= highlight_duration:
            highlight_energy = np.convolve(energy_per_second, 
                                        np.ones(highlight_duration), 
                                        mode='valid').max()
            highlight_check = highlight_energy > 0.12 * highlight_duration
            if highlight_check: score += 1
        else:
            highlight_check = "Audio too short"
    except:
        highlight_check = "Calculation error"
    
    print("\n=== Viral Trend Analysis Report ===")
    print(f"1. Tempo: {tempo:.1f} BPM ({'Good' if tempo_check else 'Outside ideal range'})")
    print(f"2. Energy: {avg_rms:.2f} ({'Strong' if energy_check else 'Weak'})")
    print(f"3. Repetition: {unique_segments} segments ({'Good' if repetition_check else 'Too varied'})")
    print(f"4. 15s Highlight: {'Found' if highlight_check is True else highlight_check}")
    print(f"\nViral Potential Score: {score}/4")
    
    return {
        'score': int(score),
        'tempo': float(tempo),
        'energy': float(avg_rms),
        'segments': int(unique_segments),
        'highlight_check': str(highlight_check)
    }

def print_acoustic_features(features):
    print("\n=== Detailed Acoustic Features ===")
    print(f"Tempo: {features['tempo']:.1f} BPM")
    print(f"Spectral Centroid: {features['spectral_centroid']:.2f} (brightness)")
    print(f"Zero Crossing Rate: {features['zero_crossing_rate']:.4f} (noisiness)")
    print(f"RMS Energy: {features['rms']:.4f} (loudness)")
    print("\nMFCC Coefficients (1-13):")
    for i in range(1, 14):
        print(f"MFCC {i}: {features[f'mfcc_{i}']:.4f}")

def generate_final_analysis(results):
    client = Groq(
        api_key="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",
    )
    
    acoustic_features = results["acoustic_features"]
    viral_analysis = results["viral_analysis"]
    
    prompt = prompt = f"""Analyze the following song data and provide a comprehensive assessment of its viral potential on a scale of 0-100, along with a detailed explanation:

=== Acoustic Features ===
- Tempo: {acoustic_features['tempo']:.1f} BPM
- Spectral Centroid (brightness): {acoustic_features['spectral_centroid']:.2f}
- Zero Crossing Rate (noisiness): {acoustic_features['zero_crossing_rate']:.4f}
- RMS Energy (loudness): {acoustic_features['rms']:.4f}

=== MFCC Analysis (Important for Vocal and Instrument Characteristics) ===
1. MFCC 1 (Brightness/High Frequency Energy): {acoustic_features['mfcc_1']:.4f} 
   - Higher values mean brighter, more present vocals
   - Lower values suggest warmer, more mellow tones

2. MFCC 2 (Sharpness/Attack Quality): {acoustic_features['mfcc_2']:.4f}
   - Higher values indicate sharper vocal articulation
   - Lower values suggest smoother vocal transitions

3. MFCC 3 (Instrument Harshness): {acoustic_features['mfcc_3']:.4f}
   - Higher values may indicate harshness in instrumentation
   - Lower values suggest smoother instrumental backing

4. MFCC 4 (Vocal Clarity): {acoustic_features['mfcc_4']:.4f}
   - Higher values mean clearer vocal presence
   - Lower values suggest more blended or distant vocals

5. MFCC 5 (Mid-Range Punch): {acoustic_features['mfcc_5']:.4f}
   - Higher values indicate stronger mid-range presence
   - Important for vocal intelligibility

6. MFCC 6 (Bass Presence): {acoustic_features['mfcc_6']:.4f}
   - Higher values mean stronger bass foundation
   - Lower values suggest a thinner low-end

7. MFCC 7 (Warmth/Low-Mid Balance): {acoustic_features['mfcc_7']:.4f}
   - Higher values indicate warmer, fuller sound
   - Important for emotional resonance

8. MFCC 8 (Metallic/Organic Balance): {acoustic_features['mfcc_8']:.4f}
   - Higher values suggest more metallic/artificial tones
   - Lower values indicate more organic/natural sound

9. MFCC 9 (Harmonic Complexity): {acoustic_features['mfcc_9']:.4f}
   - Higher values mean more complex harmonies
   - Lower values suggest simpler arrangements

10. MFCC 10 (Fundamental Bass): {acoustic_features['mfcc_10']:.4f}
    - Higher values indicate stronger fundamental frequencies
    - Affects the "fullness" of the overall sound

11. MFCC 11 (Room Acoustics): {acoustic_features['mfcc_11']:.4f}
    - Higher values suggest more reverb/room sound
    - Lower values indicate drier, more intimate recordings

12. MFCC 12 (Ultra-Low Rumble): {acoustic_features['mfcc_12']:.4f}
    - Higher values may indicate excessive sub-bass
    - Can affect streaming platform compatibility

13. MFCC 13 (Spectral Shape): {acoustic_features['mfcc_13']:.4f}
    - Overall timbral quality indicator
    - Helps identify unique sonic characteristics

=== Viral Trend Analysis ===
- Viral Potential Score: {viral_analysis['score']}/4
- Tempo Check: {'Good (110-130 BPM)' if 110 <= viral_analysis['tempo'] <= 130 else 'Outside ideal range'}
- Energy Check: {'Strong' if viral_analysis['energy'] > 0.1 else 'Weak'}
- Repetition: {viral_analysis['segments']} segments
- 15s Highlight: {viral_analysis['highlight_check']}

=== Your Task ===
1. Combine all these factors to calculate a final score between 0-100 for the song's viral potential
2. Format your response with:
   - Final Score: [0-100] within <final></final> tags
   - Summary: [2-3 paragraph detailed analysis focusing on both technical and artistic aspects]
3. Provide specific, actionable suggestions for improvement considering:
   - Vocal production adjustments based on MFCC analysis
   - Instrumentation tweaks to enhance viral appeal
   - Arrangement modifications to highlight strengths
4. Explain how current features compare to trending songs in similar genres
5. Highlight any unique sonic characteristics that could be emphasized
6. EXPLAIN IN THE OUTPUT EVERY MCFF VALUE AND WHAT IT MEANS
7. DONT GIVE ASTERISK
"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            stream=False,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating final analysis: {str(e)}"

def sigmoid_scale(score, max_streams=100000000):
    """Scale a 0-100 score to 0-100 million using a sigmoid function"""
    normalized = score / 100.0
    sigmoid = 1 / (1 + math.exp(-(10 * normalized - 5)))
    return int(sigmoid * max_streams)

def extract_final_score(analysis_text):
    """Extract the final score from <final></final> tags"""
    match = re.search(r'<final>(\d+)</final>', analysis_text)
    if match:
        return int(match.group(1))
    return 50

def analyze_song(mp3_path):
    acoustic_features = extract_acoustic_features(mp3_path)
    viral_analysis = analyze_viral_trend(mp3_path)
    
    return {
        "acoustic_features": acoustic_features,
        "viral_analysis": viral_analysis
    }

if __name__ == "__main__":
    path = select_mp3_file()
    if not path:
        print("No file selected.")
    else:
        results = analyze_song(path)
        print_acoustic_features(results["acoustic_features"])
        
        print("\n=== Final Comprehensive Analysis ===")
        final_analysis = generate_final_analysis(results)
        print(final_analysis)
        
        final_score = extract_final_score(final_analysis)
        potential_streams = sigmoid_scale(final_score)
        
        print("\n=== Viral Potential Estimate ===")
        print(f"Analysis Score: {final_score}/100")
        print(f"Potential Streams: {potential_streams:,}")
        print(f"Potential Revenue: $ {potential_streams*0.004:,}")