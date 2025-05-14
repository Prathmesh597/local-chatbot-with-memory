# D:\Github Repo\LocalChatBot\memory_manager.py

import chromadb
import json
import os
from ollama_interface import get_embedding # Import the embedding function

# --- Updated Path Configuration ---
# Get the absolute path of the directory containing this script (memory_manager.py)
# __file__ refers to the path of the current script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Since memory_manager.py is directly inside LocalChatBot, SCRIPT_DIR is the project root.
PROJECT_ROOT = SCRIPT_DIR

# Define memory paths relative to the project root
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
HISTORY_FILE_PATH = os.path.join(MEMORY_DIR, "history.jsonl")
VECTOR_DB_PATH = os.path.join(MEMORY_DIR, "vector_db")
# --- End of Updated Path Configuration ---


# Ensure memory directory exists
# These paths are now absolute based on the script's location
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)


# ChromaDB client and collection
client = None
collection = None

try:
    # Use the absolute VECTOR_DB_PATH
    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False) # Optional: disable telemetry
    )
    # Get or create the collection
    collection = client.get_or_create_collection(
        name="conversation_memory"
        # Remember: We generate embeddings externally using ollama_interface
    )
except Exception as e:
    print(f"CRITICAL Error initializing ChromaDB at {VECTOR_DB_PATH}: {e}")
    print("The chatbot memory will not function correctly. Please check ChromaDB setup and permissions.")
    # Depending on the application, might want to exit or disable memory features.


def save_conversation_turn(user_message, bot_message, turn_id):
    """
    Saves a single conversation turn (user + bot) to the history file
    and adds its embedding to the vector store.
    `turn_id` should be a unique string identifier for this turn.
    """
    if not collection:
        print("Error: ChromaDB collection not initialized. Cannot save turn.")
        return

    # Prepare the data for this turn
    turn_data = {"id": turn_id, "user": user_message, "bot": bot_message}

    # 1. Save to raw history file (using the absolute path)
    try:
        with open(HISTORY_FILE_PATH, "a", encoding="utf-8") as f: # Append mode, specify encoding
            f.write(json.dumps(turn_data) + "\n")
    except IOError as e:
        print(f"Error saving to history file {HISTORY_FILE_PATH}: {e}")
        # Consider whether to proceed with vector store saving if file save fails

    # 2. Add to vector store
    # Embed the interaction context
    text_to_embed = f"User: {user_message}\nBot: {bot_message}"
    embedding = get_embedding(text_to_embed)

    if embedding:
        try:
            collection.add(
                ids=[turn_id],
                embeddings=[embedding],
                documents=[json.dumps(turn_data)], # Store the full turn data as the document
                metadatas=[{"source": "conversation_history", "turn_id": turn_id}] # Add relevant metadata
            )
            # print(f"Added turn {turn_id} to vector store.") # Uncomment for debugging
        except Exception as e:
            # Catch potential ChromaDB specific errors if needed, e.g., DuplicateIDError
            print(f"Error adding turn_id {turn_id} to ChromaDB: {e}")
    else:
        print(f"Could not get embedding for turn_id {turn_id}. Not adding to vector store.")


def load_raw_history():
    """
    Loads the entire raw conversation history from history.jsonl using the absolute path.
    Returns a list of conversation turns (dicts).
    """
    if not os.path.exists(HISTORY_FILE_PATH):
        # print(f"History file not found: {HISTORY_FILE_PATH}") # Debug message
        return []

    history = []
    try:
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f: # Specify encoding
            for i, line in enumerate(f):
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError as jd_err:
                    print(f"Skipping malformed line {i+1} in {HISTORY_FILE_PATH}: {jd_err}")
                    continue
        return history
    except IOError as e:
        print(f"Error loading from history file {HISTORY_FILE_PATH}: {e}")
        return []


def retrieve_relevant_history(query_text, n_results=3):
    """
    Retrieves the n_results most relevant conversation turns from the vector store
    based on the query_text. Uses the initialized collection object.
    """
    if not collection:
        print("Error: ChromaDB collection not initialized. Cannot retrieve history.")
        return []

    try:
        current_collection_count = collection.count()
    except Exception as e:
        print(f"Error getting count from ChromaDB collection: {e}")
        return []

    if current_collection_count == 0:
        # print("Vector store is empty. No history to retrieve.") # Debug message
        return []

    actual_n_results = min(n_results, current_collection_count)
    if actual_n_results <= 0: # Should be caught by count check, but defense in depth
        return []

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("Could not get embedding for query. Cannot retrieve history.")
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n_results,
            include=['documents', 'metadatas', 'distances'] # Include desired data fields
        )

        retrieved_docs_json = results.get('documents', [[]])[0]
        retrieved_docs = []
        for doc_json_str in retrieved_docs_json:
            try:
                retrieved_docs.append(json.loads(doc_json_str))
            except json.JSONDecodeError as e:
                print(f"Error decoding stored document from ChromaDB: {e}. Document: {doc_json_str}")
                continue # Skip this potentially corrupted document

        # You could optionally sort retrieved_docs based on distance if needed,
        # though ChromaDB usually returns them sorted.
        # distances = results.get('distances', [[]])[0]

        return retrieved_docs
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

# --- Optional: Simple test functions ---
def _test_memory_manager():
    """ Test function for memory manager operations. """
    print("Testing Memory Manager...")
    print(f"Using History File: {HISTORY_FILE_PATH}")
    print(f"Using Vector DB Path: {VECTOR_DB_PATH}")

    # Clear previous test data if any (for repeatable tests)
    if os.path.exists(HISTORY_FILE_PATH):
        print(f"Removing old history file: {HISTORY_FILE_PATH}")
        try:
            os.remove(HISTORY_FILE_PATH)
        except OSError as e:
            print(f"Error removing history file: {e}")

    # Attempting to clear the Chroma collection
    if collection:
        try:
            print(f"Attempting to clear ChromaDB collection '{collection.name}' for test...")
            current_count = collection.count()
            if current_count > 0:
                all_ids_to_delete = collection.get(limit=current_count, include=[])['ids']
                if all_ids_to_delete:
                    print(f"Deleting {len(all_ids_to_delete)} items from ChromaDB...")
                    collection.delete(ids=all_ids_to_delete)
                print(f"ChromaDB collection '{collection.name}' cleared. Count now: {collection.count()}")
            else:
                print("ChromaDB collection is already empty.")
        except Exception as e:
            print(f"Note: Could not fully clear ChromaDB for test, or it was already clear: {e}")
            print(f"Consider manually deleting the directory {VECTOR_DB_PATH} for a completely fresh test if needed.")
    else:
        print("ChromaDB collection not available for clearing.")


    # Simulate some conversation turns
    test_turns = [
        {"user": "Test: Hello, I am looking for information on Python.", "bot": "Test: Hi! Python is a versatile programming language.", "id": "test_turn_1"},
        {"user": "Test: Tell me about its use in web development.", "bot": "Test: Python is popular in web development with frameworks like Django and Flask.", "id": "test_turn_2"},
        {"user": "Test: My name is Alex.", "bot": "Test: Nice to meet you, Alex!", "id": "test_turn_3"},
        {"user": "Test: What frameworks did you mention for web dev?", "bot": "Test: I mentioned Django and Flask for Python web development.", "id": "test_turn_4"}
    ]

    print("\n--- Saving conversation turns (test) ---")
    for turn in test_turns:
        print(f"Saving: User: {turn['user'][:30]}... | Bot: {turn['bot'][:30]}...")
        save_conversation_turn(turn["user"], turn["bot"], turn["id"])

    print(f"\nChromaDB collection count after saving: {collection.count() if collection else 'N/A'}")

    print("\n--- Loading raw history (test) ---")
    raw_history = load_raw_history()
    print(f"Loaded {len(raw_history)} turns from raw history file.")
    if raw_history:
        print("Last turn from raw_history:", raw_history[-1] if raw_history else "None")


    print("\n--- Retrieving relevant history for 'What is my name?' (test) ---")
    relevant_for_name = retrieve_relevant_history("Test: What is my name?", n_results=2)
    if relevant_for_name:
        print(f"Found {len(relevant_for_name)} relevant documents for 'What is my name?':")
        for doc in relevant_for_name:
            print(doc)
    else:
        print("No relevant documents found for 'What is my name?'.")

    print("\n--- Retrieving relevant history for 'Python web frameworks' (test) ---")
    relevant_for_web = retrieve_relevant_history("Test: Python web frameworks", n_results=2)
    if relevant_for_web:
        print(f"Found {len(relevant_for_web)} relevant documents for 'Python web frameworks':")
        for doc in relevant_for_web:
            print(doc)
    else:
        print("No relevant documents found for 'Python web frameworks'.")

    print(f"\nFinal ChromaDB collection count (test): {collection.count() if collection else 'N/A'}")


if __name__ == "__main__":
    # This block runs if you execute this file directly
    # e.g., python memory_manager.py
    # Ensure Ollama server is running with the mxbai-embed-large model for testing embeddings.
    _test_memory_manager()