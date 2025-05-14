import uuid
import time
import os # For path joining
from ollama_interface import get_chat_response
from memory_manager import save_conversation_turn, retrieve_relevant_history

# Define memory directory and history file path for chat.py as well, if needed for other purposes
# However, memory_manager.py already handles these.
# MEMORY_DIR = "memory"
# HISTORY_FILE_PATH = os.path.join(MEMORY_DIR, "history.jsonl")


def generate_turn_id():
    """Generates a unique ID for each conversation turn."""
    # Using a combination of timestamp and UUID for more robust uniqueness if desired
    # For simplicity, uuid.uuid4() is often sufficient.
    return f"turn_{int(time.time())}_{uuid.uuid4().hex[:8]}"

def format_retrieved_history_for_prompt(retrieved_docs):
    """
    Formats the list of retrieved documents (from ChromaDB)
    into a string suitable for inclusion in the LLM prompt.
    Orders by relevance (as returned by Chroma).
    """
    if not retrieved_docs:
        return ""
    
    # ChromaDB results are typically ordered by similarity (most similar first)
    # We can present them in that order, or reverse it if we prefer chronological from retrieved.
    # Let's keep the order as returned by Chroma (most relevant first).
    history_str = "Relevant past conversation snippets (most relevant first):\n---\n"
    for doc in retrieved_docs: # doc is already a dict
        history_str += f"User: {doc.get('user', 'N/A')}\nBot: {doc.get('bot', 'N/A')}\n---\n"
    return history_str

def main():
    print("Starting LocalChatBot...")
    print("This bot remembers conversations across sessions.")
    print("Type 'quit' or 'exit' to end the conversation.")
    
    # `memory_manager` initializes ChromaDB when it's imported, so it should be ready.
    # We don't need to load full raw history into chat.py's main loop unless we have a specific use for it here.
    # The RAG mechanism will pull relevant parts.

    while True:
        user_input = input("You: ").strip()
        if not user_input: # Handle empty input
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Goodbye! Your conversation has been saved.")
            break

        current_turn_id = generate_turn_id()

        # 1. Retrieve relevant history from vector store
        relevant_history_docs = retrieve_relevant_history(user_input, n_results=3)

        # 2. Format this history for the prompt
        formatted_context = format_retrieved_history_for_prompt(relevant_history_docs)
        
        # 3. Construct the full prompt for the LLM
        # This prompt engineering is crucial for good RAG performance.
        system_prompt = (
            "You are a helpful and friendly conversational AI. "
            "Your goal is to assist the user based on the current conversation and any relevant past snippets provided. "
            "If past snippets are given, use them to remember details like names, preferences, or previous topics. "
            "If no relevant past snippets are provided, or if they don't seem relevant to the current question, "
            "answer based on the current question alone."
        )
        
        final_prompt_for_llm = f"{system_prompt}\n\n"
        if formatted_context:
            final_prompt_for_llm += f"{formatted_context}\n" # formatted_context already ends with "---"
        
        # Add the current user query
        final_prompt_for_llm += f"Current conversation:\nUser: {user_input}\nBot:" # Signal LLM to generate bot's response

        # print(f"\n--- DEBUG: Prompt sent to LLM ---\n{final_prompt_for_llm}\n---------------------------------\n") # Uncomment for debugging

        # 4. Get response from LLM
        bot_response = get_chat_response(prompt_text=final_prompt_for_llm)

        if not bot_response: # Handle case where LLM fails to respond
            bot_response = "I'm sorry, I had trouble generating a response. Could you try rephrasing?"

        print(f"Bot: {bot_response}")

        # 5. Save the current turn (user_input, bot_response) to memory
        save_conversation_turn(user_input, bot_response, turn_id=current_turn_id)

    print("Chat session ended.")

if __name__ == "__main__":
    # Ensure Ollama server is running with both models.
    # memory_manager.py initializes ChromaDB when it's imported.
    main()