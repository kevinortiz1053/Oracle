# %%
import os, getpass
import pprint
os.environ['OPENAI_API_KEY'] = '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'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['TAVILY_API_KEY'] = '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'

from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# %%
import requests
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# %%
# Set up the OpenAI LLM model (ensure you have the OpenAI API key set)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define a prompt template for the agent
prompt_template = """
You are a technical project manager tasked with making an implementation plan for a client.
Here is the context you need to use to answer the query:

Context:
{context}

Question:
{question}

Answer:
"""

# The answer should be in an email format that has a summary section that summarizes the current state first and a next steps section with action items.

# Use bullet points. Use the following as an example:
# Summary of current state
# -topic 1
#     -current state of this topic
# -topic 2
#     -current state of this topic

# Next steps
#     -action item for each topic. Include who is responsible for completing the action item

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
# llm_chain = LLMChain(prompt=prompt, llm=llm) # Deprecated method
# Use the new RunnableSequence to chain the prompt and the LLM together
runnable_sequence = prompt | llm

# %%
import tiktoken

# Function to count tokens in text using tiktoken
def count_tokens(text, encoding):
    tokens = encoding.encode(text)
    return len(tokens)

# Function to make text more concise
def make_concise(text, max_tokens, encoding):
    # Count the tokens in the original text
    token_count = count_tokens(text, encoding)
    
    # If the text is already within the limit, return it as is
    if token_count <= max_tokens:
        return text, token_count
    
    # Try simplifying the text by removing unnecessary words or shortening phrases
    sentences = text.split(".")
    concise_text = ""
    
    for sentence in sentences:
        if count_tokens(concise_text + sentence + ".", encoding) <= max_tokens:
            concise_text += sentence + ". "
        else:
            break  # Stop adding sentences once we've reached the token limit
    
    # If necessary, shorten the text further
    concise_text = concise_text.strip()  # Remove trailing space
    concise_token_count = count_tokens(concise_text, encoding)
    
    return concise_text, concise_token_count

# encoding = tiktoken.get_encoding("gpt-3.5-turbo")
encoding = tiktoken.encoding_for_model("gpt-4o")
max_tokens = 15000


# %%
def fetch_web_content_with_tavily(query: str, num_results: int = 5):
    # Initialize TavilySearchResults (with your Tavily API key)
    tavily_search = TavilySearchResults(api_key="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", max_results=num_results)
    
    # Provide the tool input as a dictionary with 'query' and 'num_results'
    tool_input = {
        "query": query,
        "max_results": num_results,
        #"search_depth": "advanced",
    }
    
    # Perform the search query (use run method with tool_input)
    search_results = tavily_search.run(tool_input)
    print("len of search: ", len(search_results))

    all_content = []
    
    # Extract the content from the results
    for result in search_results:
        url = result["url"]
        print(url)
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the page content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            page_content = " ".join([para.get_text() for para in paragraphs])
            all_content.append(page_content)

    return "\n\n".join(all_content)

# %%
# q = 'How to implement fastconnect, using only sites that start with https://docs.oracle.com/en-us/iaas/Content/Network/Concepts/'
# a = fetch_web_content_with_tavily(q)
# print(a)

# %%
def query_with_web_context(state_dict):
    query = state_dict['tavily_query']
    question = state_dict['use_case']
    # Fetch content from the web using Tavily search
    web_content = fetch_web_content_with_tavily(query)
    print(f"Original token count: {count_tokens(web_content, encoding)}")
    # Get concise version of the text
    web_content, token_count = make_concise(web_content, max_tokens, encoding)
    
    # Print results
    print(f"Concise token count: {token_count}")
    # print("\nWeb_content is:")
    # print(web_content)
    
    if web_content:
        # Generate an answer using the new RunnableSequence (prompt | llm), passing the web content as context
        response = runnable_sequence.invoke({"context": web_content, "question": question})
        # return response
        state_dict['state'] = str(state_dict['state']) + str(response.content)
        state_dict['response_t'] = True
        return state_dict
    else:
        return "Could not retrieve content from the search results."

# %%
def CAE(state_dict):
    # Takes use-case as input, parses it to find the OCI services it requires, sends that to the Specialist
    print('in CAE')
    use_case = state_dict['use_case']
   
    # Takes output from Specialist and see's if any other OCI services are needed, if so, runs the logic again
    
    return state_dict

# %%
def Specialist(state_dict):
    use_case = state_dict['use_case']
    # Takes OCI service as input from CAE. Generates SQL command to query the relevant DB table
    # LLM prompt is executed using the SQL query as context, output returned to CAE
    # Start small, only base DB specialist for now, get docs for base DB
    import oracledb
    import os
    os.environ['TNS_ADMIN'] = '/Users/kcortiz/Oracle Content/Data Science/Co-scientist/Wallet_JoeHahn23ai'
    
    # Set the wallet location and connection details
    wallet_location = "/Users/kcortiz/Oracle Content/Data Science/Co-scientist/Wallet_JoeHahn23ai"
    library_location = '/Users/kcortiz/Downloads/instantclient/instantclient_23_3'
    db_username = "sairag"
    db_password = "Welcome123456789!"
    service_name = "joehahn23ai_medium"
    hostname = "adb.us-chicago-1.oraclecloud.com"
    port = "1522"

    # Configure the Oracle client
    oracledb.init_oracle_client(config_dir=wallet_location, lib_dir=library_location)

    # Create the connection string
    connection_string = f"{db_username}/{db_password}@{service_name}"

    try:
        # Connect to the database
        connection = oracledb.connect(connection_string)
        print("Successfully connected to the Oracle database!")
        
        # Create the cursor
        cursor = connection.cursor()

        # Enable DBMS_OUTPUT
        cursor.execute("BEGIN DBMS_OUTPUT.ENABLE(NULL); END;")

        # Define function get_ai_response
        cursor.execute("""create or replace FUNCTION get_ai_response(
                                p_prompt       IN VARCHAR2,
                                p_action       IN VARCHAR2 DEFAULT 'narrate',
                                p_profile_name IN VARCHAR2 DEFAULT 'RAG_PROFILE'
                            ) RETURN VARCHAR2
                            IS
                                l_response VARCHAR2(32767);
                            
                            BEGIN
                                -- Call the DBMS_CLOUD_AI.GENERATE and store the response
                                l_response := DBMS_CLOUD_AI.GENERATE(
                                                    prompt       => p_prompt,
                                                    action       => p_action,
                                                    profile_name => p_profile_name
                                                );
                            
                                -- Return the response
                                    RETURN l_response;
                            END;
                            """)
        
        # Call the function DBMS_CLOUD_AI.GENERATE above and store the response
        cursor.execute("""DECLARE
                                my_result VARCHAR2(32767);
                                p_prompt  VARCHAR2(32767);
                                p_action  VARCHAR2(32767);
                                p_profile_name VARCHAR2(32767);
                        BEGIN
                            -- Call the function and store the result
                            my_result := get_ai_response(p_prompt => :use_case, p_action => 'narrate', p_profile_name => 'RAG_PROFILE');
                            
                            -- Now you can use or print the result
                            DBMS_OUTPUT.PUT_LINE('Generated Response: ' || my_result);
                        END;""", use_case=use_case) # Bind the use_case to the :use_case bind variable

        # Use cursor.var to create a bind variable that can hold the output
        lines = cursor.arrayvar(oracledb.STRING, 32767)  # Create a bind variable for the lines
        num_lines = 1  # Specify how many lines you want to fetch from DBMS_OUTPUT
        
        # Fetch the DBMS_OUTPUT result
        cursor.execute("BEGIN DBMS_OUTPUT.GET_LINES(:lines, :num_lines); END;", 
                          lines=lines, num_lines=num_lines)
        results = lines.getvalue()

        # Print the result from DBMS_OUTPUT
        # for result in results:
        #     print(f"Output: {result}")

        
        # result = cursor.fetchone()
        # print(f"Test query result: {result}")
        
        cursor.close()
    except Exception as e:
        print (f"Connection failed: {e}")
    state_dict['state'] = state_dict['state'] + results[0]
    state_dict['response_s'] = True
    return state_dict

# %%
#Specialist('who was gatsby in love with')

# %%
def find_help():
    return None

def format_response(state_dict):
    # print('-----------------------------------------')
    # print('LLM conversation history: ', state_dict['state'])
    # p = 'Remove all the metadata at the end of this text: ' + state_dict['state']
    p = 'Only keep text relating to ' + state_dict['use_case'] + '. Text: ' + state_dict['state']
    fin_ans = llm.invoke(p)
    state_dict['final_answer'] = fin_ans.content
    print('-----------------------------------------')
    return state_dict


# %%
def CAE_criteria(state_dict):
    # print('CAE_criteria: ', state_dict)
    if state_dict['response_s'] and state_dict['response_t']:
        print('CAE criteria met for Tavily')
        return "Format Final Response"
    if state_dict['response_s']:
        print('CAE criteria met for Specialist')
        return "Query with Web Context"
    return "Specialist"
    

# %%
# Building the graph
builder = StateGraph(State)
builder.add_node("CAE", CAE)
builder.add_node("Specialist", Specialist)
builder.add_node("Query with Web Context", query_with_web_context)
# builder.add_node("Find Help", find_help) # function would need to hook up to Oracle's aria or some sort of directory. This is to find experts specializing in your question. 
builder.add_node("Format Final Response", format_response)

# Building the flow
builder.add_edge(START, "CAE")
builder.add_conditional_edges("CAE", CAE_criteria)
# builder.add_edge("CAE", "Specialist")#, data_flow="use_case")
builder.add_edge("Specialist", "CAE")
# builder.add_edge("CAE", "Query with Web Context")
builder.add_edge("Query with Web Context", "CAE")
# builder.add_edge("CAE", "Find Help")
# builder.add_edge("Find Help", "CAE")
# builder.add_edge("CAE", "Format Final Response")
builder.add_edge("Format Final Response", END)
# builder.add_edge("CAE", END)
graph = builder.compile()

# %%
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
# How my agents will be able to record information and talk to each other
class State(TypedDict):
    state: str
    notes: str
    final_answer: str
    use_case: str
    tavily_query: str
    response_s: str
    response_t: str

# %%
Original_dict = {
    'state': "",
    'notes': str,
    'final_answer': "",
    # 'use_case': 'who is gatsby in love with?',
    'use_case': 'how to create a base database in oracle cloud infrastructure console',
    # 'tavily_query': 'who is gatsby in love with?',
    'tavily_query': 'how to create a base database in oracle cloud infrastructure console, using only sites that start with https://docs.oracle.com/',
    'response_s': False,
    'response_t': False
}

a = graph.invoke(Original_dict)

print('Final Answer:')
print(a['final_answer'])

# %%
