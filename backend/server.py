import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

# =====================================================================
# ### [CHANGE 1]: FastAPI aur CORS import kiya taaki Chrome Extension
# ###             is Python file ko HTTP request bhej sake.
# =====================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS allow karna zaroori hai warna Chrome browser request block kar dega
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Extension se aane wale JSON data ka structure (video_id aur question)
class QueryRequest(BaseModel):
    video_id: str
    question: str

# Dictionary banayi taaki ek video ka FAISS index baar-baar na banana pade (Cache)
retriever_cache = {}
# =====================================================================


# Load environment variables from .env file

load_dotenv()

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise RuntimeError("Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in your environment or .env file.")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=hf_token,
)

# Yeh model (sirf ~80MB) aapke Mac par locally chalega — 0 API credits lagenge!
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = ChatHuggingFace(llm=llm)

#//step 1 ->indexing

def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# =====================================================================
# ### [CHANGE 2]: Hardcoded `video_id = "_XeE2aEiPHk"` hatakar aapke
# ###             Step-1 aur Step-2 code ko ek function `get_retriever(video_id)`
# ###             ke andar daal diya, taaki jo bhi video_id Chrome Extension
# ###             bheje, uska retriever ban jaye.
# =====================================================================
def get_retriever(video_id: str):
    # Agar is video ka FAISS vector store pehle se bana hua hai, toh seedha return kar do
    if video_id in retriever_cache:
        return retriever_cache[video_id]

    try:
        # 1. Instantiate the API class and call .fetch()
        ytt_api = YouTubeTranscriptApi()
        # ### Note: ["en", "hi", "ja"] kiya taaki English/Hindi/Japanese teeno videos chalein
        transcript_list = ytt_api.fetch(video_id, languages=["en", "hi", "ja"])
        #print(transcript_list)

        # 2. Use chunk.text (attribute) instead of chunk["text"] (dictionary key)
        transcript = " ".join(
            f"[{format_timestamp(chunk.start)}] {chunk.text}"
            for chunk in transcript_list
        )
        #print(transcript)

    except TranscriptsDisabled:
        print("No captions available for this video.")
        raise HTTPException(status_code=400, detail="No captions available for this video.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


    # transcript mil gyi ab usko chunks mein break krunga using text spilliter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # len(chunks)  # Check the number of chunks created
    # print(chunks)


    # ab split wale ko emebeg=ddin mein conver aur fir vector store mein store krunga

    vector_store = FAISS.from_documents(chunks, embeddings)

    #AGAR EMBEDDING KO DEKHNA HAi VEXTOR
    # # Get the raw numpy array of floats for chunk 0
    # total_vectors = vector_store.index.ntotal

    # # Extract all vectors from the C++ FAISS index at once
    # all_vectors = vector_store.index.reconstruct_n(0, total_vectors)

    # for i in range(total_vectors):
    #     doc_id = vector_store.index_to_docstore_id[i]
    #     doc_text = vector_store.docstore._dict[doc_id].page_content

    #     print(f"\n=== CHUNK {i} ===")
    #     print("Text preview:", doc_text[:100], "...")
    #     print("Vector length:", len(all_vectors[i]))
    #     print("Actual Vector (first 15 floats):", all_vectors[i][:15])

    #STEP-2 RETRIVAL

    # ab iss step mein user ke query ki embedding generate kr ke jo vector store mein chunks ki embedding hai
    #unse match krunga aur jo sabse zyada match hoga means relevant text or context usko le aaunga
    #cosine similarity match krte hai jo paas hota hai wo relevant
    #k ki jitni value utne relevant documents

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Cache mein save kar liya taaki agli query fast ho
    retriever_cache[video_id] = retriever
    return retriever
# =====================================================================


# ab step 4

# ab iss revelant document ko + query ko prompt bna kr llm ko denge aur response lenge

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parser = StrOutputParser()


# =====================================================================
# ### [CHANGE 3]: Hardcoded `main_chain.invoke(...)` ko `@app.post("/ask")`
# ###             endpoint ke andar rakh diya. Jab Chrome Extension `/ask`
# ###             par `video_id` aur `question` bhejega, tab yeh chalega.
# =====================================================================
@app.post("/ask")
def ask_question(req: QueryRequest):
    # Step 1 & 2: Video ID ke hisaab se retriever lao
    retriever = get_retriever(req.video_id)

    # Step 3 & 4: Aapki exact same chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | model | parser

    # User ka bheja hua question invoke karo
    answer = main_chain.invoke(req.question)
    print("\n--- ANSWER ---\n")
    print(answer)

    # Extension ko JSON mein answer wapas bhej do
    return {"answer": answer}