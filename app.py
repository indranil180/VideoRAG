import gradio as gr
from dataclasses import dataclass
from typing import List, Optional
import json
from moviepy.editor import VideoFileClip
import os
import tempfile
import lancedb
from MultimodalEmbeddings import MultimodalEmbeddings
from multimodal_lancedb import MultimodalLanceDB
from lvlm import lvlm_inference_module
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from utils import encode_image

@dataclass
class VideoQA:
    rag_chain: any
    temp_video_path: Optional[str] = None
    current_timestamp: Optional[float] = None

def get_default_rag_chain():
    LANCEDB_HOST_FILE = "lancedb"
    TBL_NAME = "test_tbl_1"
    
    db = lancedb.connect(LANCEDB_HOST_FILE)
    embedder = MultimodalEmbeddings()
    vectorstore = MultimodalLanceDB(uri=LANCEDB_HOST_FILE, embedding=embedder, table_name=TBL_NAME)
    retriever_module = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})
    
    def prompt_processing(input):
        retrieved_results, user_query = input['retrieved_results'], input['user_query']
        retrieved_result = retrieved_results[0]
        
        metadata_retrieved_video_segment = retrieved_result.metadata
        
        transcript = metadata_retrieved_video_segment['transcript']
        frame_path = metadata_retrieved_video_segment['extracted_frame_path']
        
        return {
            'prompt': f"The transcript associated with the image is '{transcript}'. {user_query}",
            'image': encode_image(frame_path),
            'metadata': metadata_retrieved_video_segment,
        }
    prompt_processing_module = RunnableLambda(prompt_processing)

    mm_rag_chain = (
        RunnableParallel({
            "retrieved_results": retriever_module,
            "user_query": RunnablePassthrough()
        })
        | prompt_processing_module
        | RunnableParallel({
            'final_text_output': lvlm_inference_module,
            'input_to_lvlm': RunnablePassthrough()
        })
    )
    return mm_rag_chain

def create_video_clip(video_path: str, start_time: float) -> str:
    """Create a new video clip starting from the timestamp"""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        with VideoFileClip(video_path) as video:
            start_time = min((float(start_time)/1000.0)-4.0, video.duration - 1)
            # start_time = (float(start_time)/1000.0)-4.0
            new_clip = video.subclip(start_time)
            new_clip.write_videofile(temp_path, codec='libx264', audio_codec='aac')
            
        return temp_path
    except Exception as e:
        print(f"Error creating video clip: {e}")
        return video_path

def cleanup_temp_file(video_qa: VideoQA):
    """Clean up temporary video file"""
    if video_qa.temp_video_path and os.path.exists(video_qa.temp_video_path):
        try:
            os.unlink(video_qa.temp_video_path)
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")

def process_query(video_qa: VideoQA, query: str) -> tuple[List[tuple[str, str]], VideoQA, str]:
    """Process user query using RAG chain and return response with timestamp"""
    cleanup_temp_file(video_qa)
    
    # Get response from RAG chain
    rag_response = video_qa.rag_chain.invoke(query)
    # Extract relevant information
    metadata = rag_response['input_to_lvlm']['metadata']
    if rag_response['final_text_output'].find("<|start_header_id|>assistant<|end_header_id|>")!= -1:
        answer = rag_response['final_text_output'].split("<|start_header_id|>assistant<|end_header_id|>")[1]
    else:
        answer = rag_response['final_text_output'].split("<|begin_of_text|>")[1]
    timestamp = metadata.get('mid_time_ms', 0.0)
    video_path = metadata.get('video_path', '')
    
    # Create video clip from timestamp
    video_qa.temp_video_path = create_video_clip(video_path, timestamp)
    video_qa.current_timestamp = timestamp
    
    response = f"Answer: {answer}\nTimestamp: {timestamp} miliseconds"
    chat_history = [[query, response]]
    
    return chat_history, video_qa, video_qa.temp_video_path

def create_demo():
    rag_chain = get_default_rag_chain()
    # video_qa = VideoQA(rag_chain=rag_chain) ## commenting the code since this line is causing error with latest gradio version
    def init_state():
        return VideoQA(rag_chain=rag_chain)

    predefined_questions = [
        "Did the speaker talked about Rajat Patidar and what is his take on his retention?",
        "Is Cam Green gonna be retained by RCB ?",
        "How many runs Faf made last year?",
    ]
    
    with gr.Blocks(theme="dark") as demo:
        gr.HTML("<h1 style='text-align: center'>Video Q&A System</h1>")
        
        with gr.Row():
            # Video player
            video = gr.Video(
                height=400,
                width=600,
                interactive=True,
                autoplay=True
            )
            
            # Chat interface
            chatbot = gr.Chatbot(height=400)
        
        # Query input
        with gr.Row():
            query = gr.Dropdown(
                choices=predefined_questions,
                allow_custom_value=True,
                label="Ask a question",
                scale=8
            )

            submit_btn = gr.Button("Ask", variant="primary", scale=1)

        state = gr.State(None)  # Initialize with None instead of VideoQA instance
        
        # Modified process_query wrapper to handle None state
        def wrapped_process_query(state, query):
            if state is None:
                state = init_state()
            return process_query(state, query)
        
        # Handle query submission
        submit_btn.click(
            wrapped_process_query,
            inputs=[state, query],
            outputs=[chatbot, state, video]
        )
        
        # Modified clear function
        def clear_fn():
            if state is not None:
                cleanup_temp_file(state)
            new_qa = init_state()
            return [], new_qa, None

        gr.Button("Clear").click(
            clear_fn,
            outputs=[chatbot, state, video]
        )
        
        # Clean up on page unload
        demo.load(lambda: None, None, None)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
