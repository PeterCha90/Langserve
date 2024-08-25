from chain import chain
from fastapi import FastAPI
from llm import llm as model
from typing import List, Union
from langserve import add_routes
from rag import chain as rag_chain
from chat import chain as chat_chain
from xionic import chain as xionic_chain
from translator import chain as EN_TO_KO_chain
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt")


add_routes(app, chain, path="/prompt")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

add_routes(app, EN_TO_KO_chain, path="/translate")

add_routes(app, rag_chain, path="/rag", enable_feedback_endpoint=True,
           enable_public_trace_link_endpoint=True,)
add_routes(app, model, path="/llm")
add_routes(
    app,
    xionic_chain.with_types(input_type=InputChat),
    path="/xionic",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
