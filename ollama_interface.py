import requests
import json

# Configuration for Ollama API
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API URL
GENERATION_MODEL_NAME = "gemma2:2b"  # <--- CHANGED HERE
EMBEDDING_MODEL_NAME = "mxbai-embed-large:335m-v1-fp16"

def get_embedding(text, model_name=EMBEDDING_MODEL_NAME):
    """
    Gets an embedding for the given text using the specified Ollama model.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model_name, "prompt": text}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from Ollama: {e}")
        return None
    except KeyError:
        print(f"Error: 'embedding' key not found in Ollama response. Full response: {response.text}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from Ollama (embeddings). Response text: {response.text}")
        return None


def get_chat_response(prompt_text, model_name=GENERATION_MODEL_NAME):
    """
    Gets a chat response from the specified Ollama model using the provided prompt_text.
    The prompt_text should be fully constructed by the caller.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt_text, # Directly use the provided prompt_text
                "stream": False
                # You can add options here if llama3.2:3b supports them, e.g.,
                # "options": {
                #     "temperature": 0.7,
                #     "num_predict": 256 # Max tokens to generate
                # }
            }
        )
        response.raise_for_status()
        
        response_data = response.json()
        return response_data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"Error getting chat response from Ollama: {e}")
        return "Sorry, I encountered an error trying to respond."
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from Ollama (chat). Response text: {response.text}")
        return "Sorry, I received an unreadable response."

# --- Optional: Simple test functions (can be commented out or removed later) ---
def _test_embedding():
    print("Testing embedding function...")
    embedding = get_embedding("Hello, world!")
    if embedding:
        print(f"Got embedding (first 5 dimensions): {embedding[:5]}")
        print(f"Embedding dimension: {len(embedding)}")
    else:
        print("Failed to get embedding.")

def _test_chat_response():
    print("\nTesting chat response function...")
    # Simple test without history formatted into the prompt
    response = get_chat_response("User: Why is the sky blue?\nBot:")
    print(f"Bot (no history): {response}")

    # Test with some history (history now needs to be part of the prompt_text)
    history_prompt = "User: My name is Prathmesh.\nBot: Nice to meet you, Prathmesh!\nUser: What is my name?\nBot:"
    response_with_history = get_chat_response(history_prompt)
    print(f"Bot (with history): {response_with_history}")


if __name__ == "__main__":
    # This block runs if you execute this file directly (e.g., python ollama_interface.py)
    # Ensure Ollama server is running with the specified models before testing.
    _test_embedding()
    _test_chat_response()