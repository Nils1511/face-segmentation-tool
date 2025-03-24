# minimal_gradio.py
import gradio as gr
import os

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Gradio on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
