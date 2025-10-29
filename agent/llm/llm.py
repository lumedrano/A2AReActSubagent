from langchain_ollama import ChatOllama

client_llama = ChatOllama(
    model="llama3",
    temperature=0.3,
    base_url="http://localhost:11434"
)

client_mistral = ChatOllama(
    model="mistral:7b",
    temperature=0.3,
    base_url="http://localhost:11434"
)