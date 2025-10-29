# react_agent_ollama.py
import asyncio
from uuid import uuid4
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Role, TextPart, AgentCard, AgentCapabilities, AgentSkill
from utils.server import run_agent_blocking
from utils.logging_utils import get_logger, Colors, configure_logging
from utils.config import SUBAGENT1_PORT
from llm.llm import client_mistral
from utils.protocol_wrappers import extract_text

# LangChain imports
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

configure_logging()
logger = get_logger(__name__)


#TODO: fix parsing of message and look at synchronization between each agents


def extract_text_from_result(result) -> str:
    messages = getattr(result, 'messages', [])  # some versions use list directly
    # Iterate backwards to find the last AI message
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            return msg.content
    return ""



# -------------------------------
# Tools
# -------------------------------
@tool
def get_weather(city: str) -> str:
    """Return weather info for a city."""
    return f"It's always sunny in {city}!"
@tool
def echo(text: str) -> str:
    """Echo the input."""
    return f"You said: {text}"

TOOLS = [get_weather, echo]

# -------------------------------
# Initialize ReAct agent
# -------------------------------
react_agent = create_agent(
    tools=TOOLS,
    model=client_mistral,
    system_prompt="You are a helpful assistant. Use tools when necessary."
)

# -------------------------------
# AgentCard
# -------------------------------
skill = AgentSkill(
    id="react_tool_user",
    name="ReAct Reasoning & Tool Use",
    description="Uses reasoning and external tools to answer questions dynamically",
    tags=["react", "tools", "reasoning"]
)

card = AgentCard(
    name="ReActAgent",
    description="A ReAct agent powered by Ollama + LangChain",
    url=f"http://localhost:{SUBAGENT1_PORT}",
    version="0.0.1",
    protocol_version="0.2.5",
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"]
)

# -------------------------------
# Executor
# -------------------------------
class ReActExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        try:
            query = extract_text(context.message)
            if not query:
                raise ValueError("No user query provided.")

            logger.info(f"{Colors.HEADER}ReAct Agent received query: {query}{Colors.ENDC}")

            # Run agent in a thread to avoid blocking the event loop
            # loop = asyncio.get_event_loop()
            result = await react_agent.ainvoke({
                "messages": [{"role": "user", "content": f"{query}"}]
            })
            print(result["messages"][-1].content)

            final_message = Message(
                message_id=f"response-msg-{uuid4().hex}",
                role=Role.agent,
                parts=[TextPart(text=result["messages"][-1].content)]
            )
            await event_queue.enqueue_event(final_message)

        except Exception as e:
            logger.error(f"Error in ReAct Agent: {e}")
            final_message = Message(
                message_id=f"error-msg-{uuid4().hex}",
                role=Role.agent,
                parts=[TextPart(text=f"Error: {e}")]
            )
            await event_queue.enqueue_event(final_message)
        finally:
            await event_queue.close()

    async def cancel(self, context, event_queue):
        pass

# -------------------------------
# Run the Agent
# -------------------------------
if __name__ == "__main__":
    run_agent_blocking(
        name="ReActAgent",
        port=SUBAGENT1_PORT,
        agent_card=card,
        executor=ReActExecutor()
    )
