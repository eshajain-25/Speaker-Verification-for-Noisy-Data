#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pydub import AudioSegment
from io import BytesIO

def convert_to_wav(input_file, target_sample_rate=16000):
    # Load the MP4 file
    audio = AudioSegment.from_file(input_file)

    # Set target sample rate
    audio = audio.set_frame_rate(target_sample_rate)

    # Export to WAV format and store it in a BytesIO object
    output_stream = BytesIO()
    audio.export(output_stream, format="wav")

    # Reset stream position to beginning
    output_stream.seek(0)

    return output_stream.read()

# Example usage:





# In[4]:


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import soundfile as sf

def clean_audio(wav_data, alpha=1):
    """
    Cleans the audio by applying spectral subtraction.

    Parameters:
    - wav_data (bytes): WAV audio data.
    - alpha (float): Spectral subtraction parameter.

    Returns:
    - io.BytesIO: Cleaned audio data.
    """
    # Load the audio data from BytesIO
    y, sr = librosa.load(io.BytesIO(wav_data), sr=None)

    # Calculate the Short-Time Fourier Transform (STFT) of the audio signal
    D = librosa.stft(y)

    # Estimate the power spectrogram (magnitude squared of the STFT)
    magnitude = np.abs(D)**2

    # Estimate the noise spectrum as the median along the time axis
    noise_spectrum = np.median(magnitude, axis=1)

    # Apply spectral subtraction to estimate the clean speech magnitude
    clean_magnitude = np.maximum(0, magnitude - alpha * noise_spectrum[:, np.newaxis])

    # Retrieve the phase information from the STFT
    phase = np.exp(1j * np.angle(D))

    # Inverse STFT to obtain the clean speech signal
    clean_signal = librosa.istft(np.sqrt(clean_magnitude) * phase)

    # Convert the clean audio signal to BytesIO
    output_stream = io.BytesIO()
    sf.write(output_stream, clean_signal, sr, format='wav')

    output_stream.seek(0)

    return output_stream.read()






# In[5]:


from pyannote.audio import Model, Inference
from pyannote.audio import Inference
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity


# In[6]:


model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")


# In[7]:


import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

def get_embeddings_from_wav(wav_data):
    with BytesIO(wav_data) as f:
        inference = Inference(model, window="whole")
        embedding = inference(f)
    return embedding



def calculate_cosine_similarities(x,y, folder_path):
  # Load the reference embedding
  embedding1=get_embeddings_from_wav(x)
  embedding1_flat = embedding1.reshape(1, -1)
  embedding2 = get_embeddings_from_wav(y)  # Replace with your inference function
  embedding2_flat = embedding2.reshape(1, -1)

  # Initialize empty lists for storing similarities and filenames
  similarities = []
  filenames = []

  # Loop through audio files in the folder
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    try:
      # Load the current embedding (handle potential errors)
      embedding= inference(file_path)
      embedding_flat = embedding.reshape(1, -1)
      arrays = [embedding1_flat, embedding2_flat,embedding_flat]
      # Calculate cosine similarity
      similarity = cosine_similarity(np.vstack(arrays))
      sim=max(similarity[0][2],similarity[1][2],similarity[2][0],similarity[2][1]);
      similarities.append(sim)
      filenames.append(filename)
    except Exception as e:
      print(f"Error processing file '{filename}': {e}")  # Informative error message

  # Find the top two highest similarity scores and their corresponding filenames
  top_two_indices = np.argsort(similarities)[::-1]  # Get indices of top two values
  top_two_sims = [similarities[top_two_indices[i]] for i in range(len(top_two_indices)) if i<2]
  top_two_filenames = [filenames[top_two_indices[i]] for i in range(len(top_two_indices)) if i<2]

  return top_two_sims, top_two_filenames

# Example usage (replace paths according to your setup)


# In[8]:


def pipeline(input_file,folder_path):
   
    # Step 1: Convert file to wav
    
    wav_data1 = convert_to_wav(input_file)
    
    # Step 2: Denoise wav file
    cleaned_audio = clean_audio(wav_data1)
    
 
    # Step 3: Calculate similarity
    top_two_sims, top_two_filenames = calculate_cosine_similarities(wav_data1,cleaned_audio, folder_path)
    print("\nTop Two Highest Similarities:")
    ans = []
    for i in range(0,2):
       ans.append((top_two_sims[i],top_two_filenames[i]))
    return ans

    




# In[9]:



# In[ ]:




