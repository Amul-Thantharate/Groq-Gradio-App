import gradio as gr
import groq
import io
import numpy as np
import soundfile as sf

def transcribe_audio(audio, api_key):
    if audio is None:
        return ""
    client = groq.Client(api_key=api_key)
    audio_data = audio[1]  
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, audio[0], format='wav')
    buffer.seek(0)
    bytes_audio = io.BytesIO()
    np.save(bytes_audio, audio_data)
    bytes_audio.seek(0)
    try:
        completion = client.audio.transcriptions.create(
            model="distil-whisper-large-v3-en",
            file=("audio.wav", buffer),
            response_format="text"
        )
        return completion
    except Exception as e:
        return f"Error in transcription: {str(e)}"

def generate_response(transcription, api_key):
    if not transcription:
        return "No transcription available. Please try speaking again."
    
    client = groq.Client(api_key=api_key)
    
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in response generation: {str(e)}"

def process_audio(audio, api_key):
    if not api_key:
        return "Please enter your Groq API key.", "API key is required."
    transcription = transcribe_audio(audio, api_key)
    response = generate_response(transcription, api_key)
    return transcription, response

custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.gr-button-primary {
    background-color: #f55036 !important;
    border-color: #f55036 !important;
}
.gr-button-secondary {
    color: #f55036 !important;
    border-color: #f55036 !important;
}
#groq-badge {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
"""

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎙️ Groq x Gradio Voice-Powered AI Assistant")
    api_key_input = gr.Textbox(type="password", label="Enter your Groq API Key")
    with gr.Row():
        audio_input = gr.Audio(label="Speak!", type="numpy")
    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription")
        response_output = gr.Textbox(label="AI Assistant Response")
    submit_button = gr.Button("Process", variant="primary")
demo.launch()