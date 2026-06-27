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


def get_completion(prompt, model="gpt-5"):
        messages = [
              {
                "role": "system",
                "content": "You are a helpful assistant designed to respond in json format."
            },
            {"role": "user", "content":prompt}]
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=0
        )
        print("\n Response: ", response, "\n")
        return response.choices[0].message.content

# Prompt the model to completion 
response = get_completion("What is the capital of France?")
print(response)

# examing tokens
response = get_completion("Take the letters in lollipop \
and reverse them")
print(response)

response = get_completion("""Take the letters in \
l-o-l-l-i-p-o-p and reverse them""")
print(response)

# count tokens
encoding = tiktoken.encoding_for_model("gpt-5")
print(encoding.encode("Hello, world!"))
print(encoding.decode(encoding.encode("Hello, world!")))

# helper function for chat format
def get_completion_from_messages(messages, model="gpt-5", temperature=0, max_tokens=500):
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("\n Response: ", response, "\n")
        return response.choices[0].message.content

# let us vary the temperature
messages =  [  
{'role':'system', 
 'content':"""You are an assistant who
 responds in the style of Dr Seuss."""},    
{'role':'user', 
 'content':"""write me a very short poem\
 about a happy carrot"""},  
] 
response = get_completion_from_messages(messages, temperature=1)
print(response)

# length
messages =  [  
{'role':'system',
 'content':'All your responses must be \
one sentence long.'},    
{'role':'user',
 'content':'write me a story about a happy carrot'},  
] 
response = get_completion_from_messages(messages, temperature =1)
print(response)

# combined  
messages =  [  
{'role':'system',
 'content':"""You are an assistant who \
responds in the style of Dr Seuss. \
All your responses must be one sentence long."""},    
{'role':'user',
 'content':"""write me a story about a happy carrot"""},
] 
response = get_completion_from_messages(messages, 
                                        temperature =1)
print(response)


# helper function for chat format
def get_completion_and_token_count(messages, model="gpt-5", temperature=0, max_tokens=500):
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        token_dict = {}
        token_dict['prompt_tokens'] = response.usage.prompt_tokens
        token_dict['completion_tokens'] = response.usage.completion_tokens
        token_dict['total_tokens'] = response.usage.total_tokens
        return content, token_dict

# let us vary the temperature
messages = [
{'role':'system', 
 'content':"""You are an assistant who responds
 in the style of Dr Seuss."""},    
{'role':'user',
 'content':"""write me a very short poem
 about a happy carrot"""},  
] 
response, token_dict = get_completion_and_token_count(messages)
print(response)
print(token_dict)
