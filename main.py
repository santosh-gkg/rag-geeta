from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.storage import InMemoryByteStore
from groq import Groq
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Load the vector database
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)
geeta = Chroma(persist_directory='geeta', embedding_function=embeddings)

app = FastAPI()

origins = [
    "http://localhost:3000",  # React app origin
    "http://localhost:5173",  # Vite default dev server port
    "http://sgkg.tech"
    "http://sgkg.tech/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# The metadata field info
metadata_field_info = [
    AttributeInfo(
        name="name",
        description='the part of the verse, it is always one of the  ["verse", "synonyms", "translation" ,"purport"]',
        type="string",
    ),
    AttributeInfo(
        name="url",
        description="The source url of the website to refer for more information",
        type="string",
    ),
    AttributeInfo(
        name="chapter",
        description="The chapter number of the bhagavad gita",
        type="integer",
    ),
    AttributeInfo(
        name="verse_number", description="the verse number of the verse", type="integer"
    ),
]
document_content_description = "The verses of the bhagavad gita along with their synonyms and translations of the verses."

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Llama3-70b-8192", temperature=0, groq_api_key=GROQ_API_KEY)

# Retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    geeta,
    document_content_description,
    metadata_field_info,
)

with open("verses_only.json") as f:
    verses_only = json.load(f)

with open("translation_only.json") as f:
    translation_only = json.load(f)


def context_generator(query):
    similarity_search = retriever.invoke(query)
    context = " \n"
    verse_urls = []
    for i in range(len(similarity_search)):
        chapter_number = similarity_search[i].metadata["chapter"]
        verse_number = similarity_search[i].metadata['verse_number']
        verse_info = f"Chapter {chapter_number} verse {verse_number} says: \n "
        verse_info += similarity_search[i].page_content + "\n"
        sanskrit_text = " the sanskrit text of this verse is : \n" + verses_only[f'{chapter_number}'][verse_number-1]
        english_translation = f" \n the english translation of the verse is :" + verses_only[f'{chapter_number}'][verse_number-1] + "\n"
        verse_info += sanskrit_text + "\n" + english_translation

        verse_url =  similarity_search[i].metadata['url']
        verse_urls.append(verse_url)
        context += verse_info
    return context, verse_urls


def prompt_generator(prompt):
    context,verse_urls = context_generator(prompt)
    statement = f"Here are the verses from bhagwad geeta : <verses> \n {context} \n <verses>"
    statement += " using these verses answer the question below, if the verses does not provide any reference to the question simply accept you are unable to answer it. "
    statement += f"<question> : \n {prompt} \n <question>"
    post = "while answering always use the sanskrit text of the verse "
    return statement, verse_urls


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]



client = Groq(api_key=GROQ_API_KEY)


@app.post("/geeta/")
async def chat(request: ChatRequest):
    messages = request.messages
    user_message = messages[-1].content
    # return {"response": messages}

    def parse_groq_stream(stream):
        response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response
    newprompt,sources=prompt_generator(user_message)
    stream = client.chat.completions.create(
        model="Llama3-8b-8192",
        messages=[
            {"role": "system", "content": "you are hari kirtan dasa a iskcon devotee. you always greet with 'hare krishna' and end with  srila prabhupada ki jai 'your servant hari kirtana dasa'"},
        ] +
        [
            {"role": m.role, "content": m.content}
            for m in messages[:-1]
        ] + [{"role": "user", "content": newprompt}],
        temperature=0.0,
        # stream=True,
    )

    response = stream.choices[0].message.content
    response += "\n sources: <br>"
    sources=list(set(sources))
    for source in sources:
        response += " <a href=" + source + " target='_blank'>"+source+"</a> \n"


    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
