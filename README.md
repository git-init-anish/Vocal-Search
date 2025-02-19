# Vocal-Search

This project is a Speech-Based AI Search Engine that enables users to ask questions via voice input. The system transcribes speech, retrieves relevant information from the web using RAG (Retrieval-Augmented Generation), and generates a concise summary using a large language model. It integrates speech-to-text processing, web scraping, document retrieval, and LLM-based summarization to provide an AI-driven question-answering experience.

Core Features

1) Converts voice input into text using Whisper.

2) Enhances audio quality by noise reduction and silence removal.

3) Web Scraping & Search Engine Integration based ona Google Search

4) Extracts webpage content using Selenium.

5) Retrieves the most relevant documents using FAISS.

6) Processes the retrieved documents and generates a summarized response.


Libraries & Frameworks

LangChain: Orchestrates LLM-based responses.
Ollama (Mistral model): Generates AI-powered responses.
Whisper: Speech recognition for transcribing voice input.
Librosa & Noisereduce: Audio processing and noise reduction.
Pydub: Silence removal from audio.
Selenium: Web scraping for content extraction.
Google Custom Search API: Fetches relevant web links.
FAISS: Vector search for retrieving relevant documents.
Sentence-Transformers: Embedding text for similarity search.


Future Scope:
Future Enhancements
1)Real-time voice query processing using a live microphone input.
2)Multi-language support for speech recognition and summarization.
3)Mobile or Web UI integration for a better user experience.


Contributions are welcome! Feel free to open issues and submit pull requests.
