import chromadb
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


SYSTEM_MESSAGE = """You are a CV chatbot for Alexander Weyhe.
        
        You will receive retrieved context from a knowledge base.
        
        Instructions:
        -Answer in the language of the question asked.
        -Answer only using information contained in the retrieved context.
        -If the context is insufficient, do not fabricate an answer. Instead, respond naturally and vary your wording. Avoid repeating the same standard phrase every time.
        -When the requested information is unavailable, use a short, natural, and occasionally playful response appropriate to the question. For example, in German: "Diese Frage beantwortet Alex gerne persönlich.", "Diese Info liegt mir im Moment nicht vor.", "Danke für deine Frage, im Moment kann ich sie leider nicht beantworten.", "Du bist ganz schön neugierig – das weiß ich leider nicht.", "Dieses Detail ist mir nicht bekannt.", "Gute Frage, die Info habe ich gerade nicht parat.", or "Das weiß ich leider nicht."
        -In English, use equivalent natural and varied formulations, for example: "Alex would be happy to answer that personally.", "I don't have that information at the moment.", "Thanks for your question, unfortunately I can't answer that right now.", "You're quite curious – I don't know that one.", "I'm not aware of that detail.", "Good question, I don't have that information at hand.", or "Unfortunately, I don't know."
        -Choose the response that fits the tone and context of the question. Keep it concise and do not invent information.
        -Do not fabricate details.
        -Keep answers concise and directly answer the question.
        
        Answer the user's question using only the retrieved context below. 
        Also consider the message history to determine if a question refers to a previously answered question.
        
        Retrieved context:
        {context}"""


def load_index():
    embedder = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )

    db = chromadb.PersistentClient(path="./index_storage")

    chroma_collection = db.get_collection("cv_bot_index")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embedder,
    )
    
    return index


def answer_question(prompt: str, 
                    client: OpenAI, 
                    chat_history: list[dict], 
                    retriever):

    nodes = retriever.retrieve(prompt)
    context = "\n\n".join([nodes.get_content() for nodes in nodes])
    
    messages = [
        {
            "role": "developer", 
            "content": SYSTEM_MESSAGE.format(context=context),
        },
        *chat_history,
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    
    answer = response.choices[0].message.content
    
    chat_history.extend(
        [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": answer,
            }
        ]
    )
    
    return answer
