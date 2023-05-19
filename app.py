from langchain.chains import LLMChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import SeleniumURLLoader
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import pathlib
import subprocess
import tempfile

from dotenv import load_dotenv
import os
load_dotenv()
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

prompt_template = """Use the context regarding job posting below to write a cover letter cold-email expressing interest in the position from Ali:
    posting: {posting}
    role: Software Engineering Intern
    Cover Letter:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["posting"]
)

llm = OpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=PROMPT)

urls = [
    # "https://www.janestreet.com/join-jane-street/position/6603079002/",
    "https://www.jobillico.com/en/job-offer/advanced-micro-devices-inckIvlOh/software-development-engineer-2/10433906?utm_campaign=google_jobs_apply&utm_source=google_jobs_apply&utm_medium=organic"
]


loader = SeleniumURLLoader(urls=urls)
index = VectorstoreIndexCreator().from_loaders([loader])

# index_creator = VectorstoreIndexCreator(
#     vectorstore_cls=Chroma,
#     embedding=OpenAIEmbeddings(),
#     text_splitter=CharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=100, chunk_overlap=0)
# )

docs = index.vectorstore.similarity_search("Software Engineering Intern", k=1)
inputs = [{"posting": doc.page_content} for doc in docs]

emails = chain.apply(inputs)

for e in emails:
    print(e['text'])

# print(llm)
# search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings())
