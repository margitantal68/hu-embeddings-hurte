import ollama
import openai
import os 
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")



class SentenceTransformerEmbedder:
    def __init__(self, model_name="NYTK/sentence-transformers-experimental-hubert-hungarian", normalize_embeddings=True):
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=None):
        if isinstance(texts, str):
            texts = [texts]

        normalize = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize
        )
        return embeddings
    

class OllamaEmbedder:
    def __init__(self, model_name="nomic_embed_text"):
        self.model_name = model_name
        self.ollama = ollama

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            response = self.ollama.embeddings(model=self.model_name, prompt=text)
            emb = response["embedding"]
            embeddings.append(emb)
        arr = np.array(embeddings, dtype=np.float32)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr
        

class OpenAIEmbedder:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        
        embeddings = [e.embedding for e in response.data]
        arr = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        return arr


# Adapted for HuRTE - max 100 texts per embed_content call
# This is a workaround for the Gemini API limit of 100 texts per call.
# If you have more than 100 texts, you should split them into chunks.


def batch_texts(texts, batch_size=100):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

class GeminiEmbedder:
    def __init__(self, model_name="models/embedding-001"):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        client = genai.Client(api_key=GEMINI_API_KEY)
        arr = []
        if len(texts) <= 100:
            response = client.models.embed_content(
            model='models/embedding-001',
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768, task_type="retrieval_query"))

            arr = np.vstack([emb.values for emb in response.embeddings])

        else:
            for batch in batch_texts(texts, batch_size=100):
                response = client.models.embed_content(
                    model='models/embedding-001',
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=768,
                        task_type="retrieval_query"
                    )
                )
                arr.extend([emb.values for emb in response.embeddings])
            arr = np.vstack(arr)
        
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        return arr


