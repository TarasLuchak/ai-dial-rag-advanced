import os

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation (RAG) assistant that helps users with
questions about a specific microwave oven based strictly on its user manual.

Conversation messages you receive will contain a synthetic user message that
includes two sections:
1) "RAG Context" – relevant excerpts from the microwave manual
2) "User Question" – the actual question from the user

Instructions:
- Always treat the RAG Context and conversation history as the primary sources of truth.
- Use only information that is clearly supported by the RAG Context when answering.
- If the question is not related to microwave usage, installation, safety, maintenance,
  troubleshooting or other topics covered by the manual, politely refuse to answer and
  ask the user to focus on questions about the microwave manual.
- If the answer cannot be found in the RAG Context, clearly state that the manual does
  not provide enough information and do not invent details.
- Be concise, factual and helpful. When useful, briefly refer back to the relevant part
  of the context instead of quoting large chunks verbatim.
"""

# Structured user prompt, with RAG Context and User Question sections.
USER_PROMPT = """
RAG Context:
{context}

User Question:
{question}

Using only the information in the RAG Context and the existing conversation
history, provide a clear and helpful answer to the User Question. If the
answer is not supported by the RAG Context, say that the manual does not
specify the answer and avoid speculation.
"""


EMBEDDINGS_MODEL = "text-embedding-3-small-1"
# Default chat model deployment; adjust if your deployment name is different.
CHAT_MODEL = os.getenv("DIAL_CHAT_MODEL", "gpt-4.1-mini-1")

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "vectordb",
    "user": "postgres",
    "password": "postgres",
}

embeddings_client = DialEmbeddingsClient(EMBEDDINGS_MODEL, API_KEY)
chat_client = DialChatCompletionClient(CHAT_MODEL, API_KEY)
text_processor = TextProcessor(embeddings_client, DB_CONFIG)


def _build_knowledge_base(dimensions: int = 1536) -> None:
    """
    Process the microwave manual and populate the vectors table.
    """
    base_dir = os.path.dirname(__file__)
    manual_path = os.path.join(base_dir, "embeddings", "microwave_manual.txt")

    text_processor.process_text_file(
        file_path=manual_path,
        document_name="microwave_manual",
        chunk_size=300,
        overlap=40,
        dimensions=dimensions,
        truncate_table=True,
    )


def run_console_chat() -> None:
    """
    Run a simple console chat implementing the full RAG loop:
    Retrieval -> Augmentation -> Generation.
    """
    if not API_KEY or API_KEY.strip() == "":
        raise ValueError("DIAL API key is not set. Please configure `DIAL_API_KEY` env variable.")

    # Ensure vectors table is populated
    _build_knowledge_base()

    conversation = Conversation()
    # Seed conversation with the system message
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("RAG Microwave Assistant is ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # 🔍 Retrieval
        search_results = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=5,
            min_score=0.5,
            dimensions=1536,
        )
        context_chunks = [row["text"] for row in search_results] if search_results else []
        rag_context = "\n\n".join(context_chunks) if context_chunks else "No relevant context retrieved."

        # 🔗 Augmentation
        user_content = USER_PROMPT.format(context=rag_context, question=user_input)
        user_message = Message(Role.USER, user_content)
        conversation.add_message(user_message)

        # 🤖 Generation
        messages = conversation.get_messages()
        ai_message = chat_client.get_completion(messages, temperature=0.2)
        conversation.add_message(ai_message)

        print(f"Assistant: {ai_message.content}\n")


if __name__ == "__main__":
    # PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
    # RUN docker-compose.yml before starting this script.
    run_console_chat()