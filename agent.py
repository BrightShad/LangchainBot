import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# 1. Load Environment Variables
# Your .env must have: 
# DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
# GOOGLE_API_KEY=your_gemini_api_key
load_dotenv()

# 2. Database Connection
# Restricting to specific tables saves tokens and prevents hitting limits
db = SQLDatabase.from_uri(
    os.getenv("DATABASE_URL"),
    include_tables=['students'],  # Update this to your table names
    sample_rows_in_table_info=2
)

# 3. Gemini LLM Setup (Free Tier)
# gemini-1.5-flash is highly recommended for speed and higher rate limits
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    convert_system_message_to_human=True
)

# 4. Agent Instructions
# Focus on READ-ONLY behavior and displaying contents
system_message = """
You are a PostgreSQL expert designed to read and display table data.
Rules:
- Always use 'SELECT * FROM table_name LIMIT 10' when asked to show contents.
- Do not attempt to modify, delete, or drop any data.
- If the user asks for a table that doesn't exist, list the available tables first.
- do not answer in signature, always respond properly
"""

# 5. Create the SQL Agent
agent_executor = create_sql_agent(
    llm, 
    db=db, 
    agent_type="tool-calling",
    verbose=False,
    prefix=system_message,
    allow_dangerous_requests=True # Required for SQL execution
)

def start_agent():
    print("\n--- Gemini Postgres Reader is Live ---")
    print("Commands: 'exit' to quit | 'show [table_name]' to see data")
    
    while True:
        try:
            user_input = input("\nEnter your request: > ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Closing...")
                break
                
            if not user_input.strip():
                continue

            # Run the agent
            response = agent_executor.invoke({"input": user_input})
            # Extract the actual answer text
            final_answer = response.get('output', '')
            
       
            if isinstance(final_answer, list) and len(final_answer) > 0:
                clean_text = final_answer[0].get('text', 'No text found')
            else:
                clean_text = final_answer

            print(f"\nResult: {clean_text}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    start_agent()