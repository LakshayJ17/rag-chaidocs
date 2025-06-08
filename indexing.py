from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

url = [
    'https://docs.chaicode.com/youtube/getting-started/',
    'https://docs.chaicode.com/youtube/chai-aur-html/welcome/',
    'https://docs.chaicode.com/youtube/chai-aur-html/introduction/',
    'https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/',
    'https://docs.chaicode.com/youtube/chai-aur-html/html-tags/',
    'https://docs.chaicode.com/youtube/chai-aur-git/welcome/',
    'https://docs.chaicode.com/youtube/chai-aur-git/introduction/',
    'https://docs.chaicode.com/youtube/chai-aur-git/terminology/',
    'https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/',
    'https://docs.chaicode.com/youtube/chai-aur-git/branches/',
    'https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/',
    'https://docs.chaicode.com/youtube/chai-aur-git/managing-history/',
    'https://docs.chaicode.com/youtube/chai-aur-git/github/',
    'https://docs.chaicode.com/youtube/chai-aur-c/welcome/',
    'https://docs.chaicode.com/youtube/chai-aur-c/introduction/',
    'https://docs.chaicode.com/youtube/chai-aur-c/hello-world/',
    'https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/',
    'https://docs.chaicode.com/youtube/chai-aur-c/data-types/',
    'https://docs.chaicode.com/youtube/chai-aur-c/operators/',
    'https://docs.chaicode.com/youtube/chai-aur-c/control-flow/',
    'https://docs.chaicode.com/youtube/chai-aur-c/loops/',
    'https://docs.chaicode.com/youtube/chai-aur-c/functions/',
    'https://docs.chaicode.com/youtube/chai-aur-django/welcome/',
    'https://docs.chaicode.com/youtube/chai-aur-django/getting-started/',
    'https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/',
    'https://docs.chaicode.com/youtube/chai-aur-django/tailwind/',
    'https://docs.chaicode.com/youtube/chai-aur-django/models/',
    'https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/welcome/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/introduction/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/postgres/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/normalization/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/',
    'https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/welcome/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/',
    'https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/'
]


loader = WebBaseLoader(
    url
)

docs = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents=docs)

# Embedding
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    openai_api_key=openai_api_key
)

# Storing to vector DB
pc = PineconeClient(api_key=pinecone_api_key)

index_name = "chaidocs"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
    pinecone_index,
    embeddings,
    text_key="text"
)

batch_size = 100  
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    vectorstore.add_documents(batch)

print("All documents embedded and stored successfully in Pinecone.")