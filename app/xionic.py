import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("XIONIC_BASE_URL", "http://sionic.chat:8001/v1"),
    api_key=os.getenv("XIONIC_API_KEY", "your-api-key-here"),
    model=os.getenv("XIONIC_MODEL", "xionic-ko-llama-3-70b"),
)

# Prompt 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, smart, kind, and efficient AI assistant named '테디'. You always fulfill the user's requests to the best of your ability. You must generate an answer in Korean.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm | StrOutputParser()
