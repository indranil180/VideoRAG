# RAG for Video Retrieval

This project explores the implementation of Retrieval-Augmented Generation (RAG) for video data retrieval using open-source libraries and models.

## Challenges and Goals

The project tackles the challenge of applying RAG to a non-standard domain - video data. This involved data pre-processing, multimodal embedding generation, and retrieval based on user queries.

## Technical Breakdown

1. Data Creation & Transformation
* Transcription:
  * YouTube videos: Transcripts are directly used.
  * Local videos (.mp4): OpenAI Whisper model is used to generate VTT files for videos lacking built-in transcripts.
  * Context Extraction: Frames are extracted from each video segment's median timestamp to provide contextual information.
2. Embedding & Storage
  * Embedding Generation: CLIP library is used to generate embeddings for both video frames and text data.
  * Data Storage: LanceDB is chosen for efficient storage of the generated embeddings.
3. Retrieval & UI
  * Multimodal RAG System: LangChain is used to build a multimodal RAG system for retrieving relevant video segments based on user queries.
  * UI Interface: Gradio provides a user-friendly interface for interacting with the system. Users can input queries and the system will identify the matching segment and play the video from the correct timestamp.
## Technology Stack

OpenAI Whisper,
CLIP,
LanceDB,
LangChain,
Gradio,
Llama 3.2 11b Vision

## License

This project is licensed under the Apache 2.0 License.
