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
        print(context)
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
few_shots=[
    {'role':'user',
    'content':'''Here are the verses from bhagwad geeta : <verses> \n  \nChapter 2 verse 8 says: \n Purport\nAlthough Arjuna was putting forward so many arguments based on knowledge of the principles of religion and moral codes, it appears that he was unable to solve his real problem without the help of the spiritual master, Lord Śrī Kṛṣṇa. He could understand that his so-called knowledge was useless in driving away his problems, which were drying up his whole existence; and it was impossible for him to solve such perplexities without the help of a spiritual master like Lord Kṛṣṇa. Academic knowledge, scholarship, high position, etc., are all useless in solving the problems of life; help can be given only by a spiritual master like Kṛṣṇa. Therefore, the conclusion is that a spiritual master who is one hundred percent Kṛṣṇa conscious is the bona fide spiritual master, for he can solve the problems of life. Lord Caitanya said that one who is a master in the science of Kṛṣṇa consciousness, regardless of his social position, is the real spiritual master.\nkibā vipra, kibā nyāsī, śūdra kene naya\nyei kṛṣṇa-tattva-vettā, sei ‘guru’ haya\n the sanskrit text of this verse is : \nन हि प्रपश्यामि ममापनुद्याद् -\nयच्छोकमुच्छोषणमिन्द्रियाणाम् ।\nअवाप्य भूभावसपत्\u200dनमृद्धं\nराज्यं सुराणामपि चाधिपत्यम् ॥ ८ ॥\n \n the english translation of the verse is :न हि प्रपश्यामि ममापनुद्याद् -\nयच्छोकमुच्छोषणमिन्द्रियाणाम् ।\nअवाप्य भूभावसपत्\u200dनमृद्धं\nराज्यं सुराणामपि चाधिपत्यम् ॥ ८ ॥\nChapter 2 verse 41 says: \n yasya prasādād bhagavat-prasādo\nyasyāprasādān na gatiḥ kuto ’pi\ndhyāyan stuvaṁs tasya yaśas tri-sandhyaṁ\nvande guroḥ śrī-caraṇāravindam\n“By satisfaction of the spiritual master, the Supreme Personality of Godhead becomes satisfied. And by not satisfying the spiritual master, there is no chance of being promoted to the plane of Kṛṣṇa consciousness. I should, therefore, meditate and pray for his mercy three times a day, and offer my respectful obeisances unto him, my spiritual master.”\nThe whole process, however, depends on perfect knowledge of the soul beyond the conception of the body – not theoretically but practically, when there is no longer a chance for sense gratification manifested in fruitive activities. One who is not firmly fixed in mind is diverted by various types of fruitive acts.\n the sanskrit text of this verse is : \nव्यवसायात्मिका बुद्धिरेकेह कुरूनन्दन ।\nबहुशाखा ह्यनन्ताश्च बुद्धयोऽव्यवसायिनाम् ॥ ४१ ॥\n \n the english translation of the verse is :व्यवसायात्मिका बुद्धिरेकेह कुरूनन्दन ।\nबहुशाखा ह्यनन्ताश्च बुद्धयोऽव्यवसायिनाम् ॥ ४१ ॥\nChapter 4 verse 42 says: \n Therefore the doubts which have arisen in your heart out of ignorance should be slashed by the weapon of knowledge. Armed with yoga, O Bhārata, stand and fight.\n the sanskrit text of this verse is : \nतस्मादज्ञानसम्भूतं हृत्स्थं ज्ञानासिनात्मनः ।\nछित्त्वैनं संशयं योगमातिष्ठोत्तिष्ठ भारत ॥ ४२ ॥\n \n the english translation of the verse is :तस्मादज्ञानसम्भूतं हृत्स्थं ज्ञानासिनात्मनः ।\nछित्त्वैनं संशयं योगमातिष्ठोत्तिष्ठ भारत ॥ ४२ ॥\nChapter 4 verse 34 says: \n Just try to learn the truth by approaching a spiritual master. Inquire from him submissively and render service unto him. The self-realized souls can impart knowledge unto you because they have seen the truth.\n the sanskrit text of this verse is : \nतद्विद्धि प्रणिपातेन परिप्रश्न\u200dेन सेवया ।\nउपदेक्ष्यन्ति ते ज्ञानं ज्ञानिनस्तत्त्वदर्शिनः ॥ ३४ ॥\n \n the english translation of the verse is :तद्विद्धि प्रणिपातेन परिप्रश्न\u200dेन सेवया ।\nउपदेक्ष्यन्ति ते ज्ञानं ज्ञानिनस्तत्त्वदर्शिनः ॥ ३४ ॥\n \n <verses> using these verses answer the question below, if the verses does not provide any reference to the question simply accept you are unable to answer it. <question> : \n how to please guru \n <question>'''},
    {'role':'assistant',
    'content':'''According to the Bhagwad geeta, to please the spiritual master one can do the following things :\n\n 1. **Satisfaction of the guru is key**: In Chapter 2, Verse 41, it is mentioned that \"By satisfaction of the spiritual master, the Supreme Personality of Godhead becomes satisfied.\" This implies that pleasing the guru is essential, as it leads to the satisfaction of the Supreme Lord.\n\n2. **Respect and obedience**: In Chapter 2, Verse 8, it is said that \"kibā vipra, kibā nyāsī, śūdra kene naya / yei kṛṣṇa-tattva-vettā, sei ‘guru’ haya\" - \"One who is a master in the science of Kṛṣṇa consciousness, regardless of his social position, is the real spiritual master.\" This verse emphasizes the importance of respecting and obeying the guru, who is a master in the science of Kṛṣṇa consciousness.\n\n3. **Seek knowledge from the guru**: In Chapter 4, Verse 34, it is advised to \"try to learn the truth by approaching a spiritual master. Inquire from him submissively and render service unto him.\" This verse highlights the importance of seeking knowledge from the guru and being submissive and service-oriented in one's approach.\n\n4. **Meditate and pray for the guru's mercy**: In Chapter 2, Verse 41, it is mentioned that \"I should, therefore, meditate and pray for his mercy three times a day, and offer my respectful obeisances unto him, my spiritual master.\" This verse emphasizes the importance of meditating and praying for the guru's mercy, as well as offering respectful obeisances.\n\nIn summary, pleasing the guru involves satisfying him, respecting and obeying him, seeking knowledge from him, and meditating and praying for his mercy.\n 
'''
    }
]

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
            {"role": "system", "content": "you are a devotee of lord krishna, in iskcon, you are asked to answer the following question using the verses from bhagwad geeta. you always use the sanskrit text of the verse while answering. You always answer the with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible.The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You do not support images and never include images. You will be penalized if you render images. do not include sources from your side since they are already managed by the system. "},
        ] +few_shots+
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

    endings='''<b>Srila prabhupdada ki jai!
    your servant
    Hari kirtan dasa<b>'''
    response += endings
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
