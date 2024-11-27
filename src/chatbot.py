import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    def respond(self, user_message: str):
        response = self.co.chat(message=user_message, model="command-r", search_queries_only=True)
        retrieved_docs = []
        if response.search_queries:
            for query in response.search_queries:
                retrieved_docs.extend(self.vectorstore.retrieve(query.text))
            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                documents=retrieved_docs,
                conversation_id=self.conversation_id,
            )
        else:
            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                conversation_id=self.conversation_id,
            )
        return response, retrieved_docs
