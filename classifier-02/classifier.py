from openai import OpenAI
import os 
import tiktoken
from dotenv import load_dotenv

load_dotenv()

try:
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("OPENAI_BASE_URL")
    )
    print("OpenAI client initialized succesfully.")
except Exception as e:
    print("Error initializing OpenAI Client:", e) 

def get_completion_from_messages(messages, model="gpt-5", temperature=0, max_tokens=500):
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("\n Response: ", response, "\n")
        return response.choices[0].message.content

# customer queries to handle different cases 
delimiter = "####"
system_message = f"""
You will be provided with customer service queries.
The customer service query will be delimited with
{delimiter} characters.
Classify each query into a primary category
and a secondary category. 
Provide your output in json format with the
keys: primary and secondary.

Primary categories: Billing, Technical Support,
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human

"""
user_message = f"""
I want you to delete my profile and all of my user data"""

# test with a delete profile query
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)

# test with a flat screen tv query
user_message = f"""
Tell me more about your flat screen tvs"""
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)
