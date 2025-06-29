from llm import llm
from graph import graph

# Create a movie chat chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()
#This creates a processing chain where: 1. User input → goes into chat_prompt
# 2. Formatted prompt → goes to llm
# 3. LLM output → goes to StrOutputParser
# 4. Final output → clean text response


# Create a set of tools
from langchain.tools import Tool

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    )
]

# Create chat history callback
from langchain_neo4j import Neo4jChatMessageHistory

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

#LangChain Hub: A repository of pre-made prompt templates
agent_prompt = hub.pull("hwchase17/react-chat")

#create_react_agent(): Creates an agent using the ReAct framework
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True #shows the agent's thinking process
    )

# RunnableWithMessageHistory: Combines the agent with chat history management
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
from utils import get_session_id

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']