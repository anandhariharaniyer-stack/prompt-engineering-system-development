import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load API credentials from the project root .env file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    print("OpenAI client initialized succesfully.")
except Exception as e:
    print("Error initializing OpenAI Client:", e)


def get_completion_from_messages(
    messages, model="gpt-5", temperature=0, max_tokens=5000
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not response.choices:
        raise ValueError(
            f"No choices returned from the API. Full response: {response}"
        )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(
            "Empty model response. With gpt-5, increase max_tokens so "
            "reasoning tokens do not consume the entire budget."
        )
    return content


def get_products_and_category():
    """Product catalog used by the category/product extraction prompts."""
    return {
        "Computers and Laptops": [
            "TechPro Ultrabook",
            "BlueWave Gaming Laptop",
            "PowerLite Convertible",
            "TechPro Desktop",
            "BlueWave Chromebook",
        ],
        "Smartphones and Accessories": [
            "SmartX ProPhone",
            "MobiTech PowerCase",
            "SmartX MiniPhone",
            "MobiTech Wireless Charger",
            "SmartX EarBuds",
        ],
        "Televisions and Home Theater Systems": [
            "CineView 4K TV",
            "SoundMax Home Theater",
            "CineView 8K TV",
            "SoundMax Soundbar",
            "CineView OLED TV",
        ],
        "Gaming Consoles and Accessories": [
            "GameSphere X",
            "ProGamer Controller",
            "GameSphere Y",
            "ProGamer Racing Wheel",
            "GameSphere VR Headset",
        ],
        "Audio Equipment": [
            "AudioPhonic Noise-Canceling Headphones",
            "WaveSound Bluetooth Speaker",
            "AudioPhonic True Wireless Earbuds",
            "WaveSound Soundbar",
            "AudioPhonic Turntable",
        ],
        "Cameras and Camcorders": [
            "FotoSnap DSLR Camera",
            "ActionCam 4K",
            "FotoSnap Mirrorless Camera",
            "ZoomMaster Camcorder",
            "FotoSnap Instant Camera",
        ],
    }


def find_category_and_product_v1(user_input, products_and_category):
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    AND
        'products': <a list of products that must be found in the allowed products below>


    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.


    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}


    """

    few_shot_user_1 = """I want the most expensive computer."""
    few_shot_assistant_1 = """
    [{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{few_shot_user_1}{delimiter}"},
        {"role": "assistant", "content": few_shot_assistant_1},
        {"role": "user", "content": f"{delimiter}{user_input}{delimiter}"},
    ]
    return get_completion_from_messages(messages)


def find_category_and_product_v2(user_input, products_and_category):
    """
    Improved prompt:
    - Require JSON-only output
    - Add a second few-shot example for cheapest computer
    """
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    AND
        'products': <a list of products that must be found in the allowed products below>
    Do not output any additional text that is not in JSON format.
    Do not write any explanatory text after outputting the requested JSON.


    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.


    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}


    """

    few_shot_user_1 = """I want the most expensive computer. What do you recommend?"""
    few_shot_assistant_1 = """
    [{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """

    few_shot_user_2 = """I want the most cheapest computer. What do you recommend?"""
    few_shot_assistant_2 = """
    [{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{few_shot_user_1}{delimiter}"},
        {"role": "assistant", "content": few_shot_assistant_1},
        {"role": "user", "content": f"{delimiter}{few_shot_user_2}{delimiter}"},
        {"role": "assistant", "content": few_shot_assistant_2},
        {"role": "user", "content": f"{delimiter}{user_input}{delimiter}"},
    ]
    return get_completion_from_messages(messages)


def eval_response_with_ideal(response, ideal, debug=False):
    if debug:
        print("response")
        print(response)

    json_like_str = response.replace("'", '"')
    l_of_d = json.loads(json_like_str)

    if l_of_d == [] and ideal == []:
        return 1
    elif l_of_d == [] or ideal == []:
        return 0

    correct = 0

    if debug:
        print("l_of_d is")
        print(l_of_d)

    for d in l_of_d:
        cat = d.get("category")
        prod_l = d.get("products")
        if cat and prod_l:
            prod_set = set(prod_l)
            ideal_cat = ideal.get(cat)
            if ideal_cat:
                prod_set_ideal = set(ideal.get(cat))
            else:
                if debug:
                    print(f"did not find category {cat} in ideal")
                    print(f"ideal: {ideal}")
                continue

            if debug:
                print("prod_set\n", prod_set)
                print()
                print("prod_set_ideal\n", prod_set_ideal)

            if prod_set == prod_set_ideal:
                if debug:
                    print("correct")
                correct += 1
            else:
                print("incorrect")
                print(f"prod_set: {prod_set}")
                print(f"prod_set_ideal: {prod_set_ideal}")
                if prod_set <= prod_set_ideal:
                    print("response is a subset of the ideal answer")
                elif prod_set >= prod_set_ideal:
                    print("response is a superset of the ideal answer")

    return correct / len(l_of_d)


msg_ideal_pairs_set = [
    {
        "customer_msg": """Which TV can I buy if I'm on a budget?""",
        "ideal_answer": {
            "Televisions and Home Theater Systems": set(
                [
                    "CineView 4K TV",
                    "SoundMax Home Theater",
                    "CineView 8K TV",
                    "SoundMax Soundbar",
                    "CineView OLED TV",
                ]
            )
        },
    },
    {
        "customer_msg": """I need a charger for my smartphone""",
        "ideal_answer": {
            "Smartphones and Accessories": set(
                [
                    "MobiTech PowerCase",
                    "MobiTech Wireless Charger",
                    "SmartX EarBuds",
                ]
            )
        },
    },
    {
        "customer_msg": """What computers do you have?""",
        "ideal_answer": {
            "Computers and Laptops": set(
                [
                    "TechPro Ultrabook",
                    "BlueWave Gaming Laptop",
                    "PowerLite Convertible",
                    "TechPro Desktop",
                    "BlueWave Chromebook",
                ]
            )
        },
    },
    {
        "customer_msg": """tell me about the smartx pro phone and \
    the fotosnap camera, the dslr one.\
    Also, what TVs do you have?""",
        "ideal_answer": {
            "Smartphones and Accessories": set(["SmartX ProPhone"]),
            "Cameras and Camcorders": set(["FotoSnap DSLR Camera"]),
            "Televisions and Home Theater Systems": set(
                [
                    "CineView 4K TV",
                    "SoundMax Home Theater",
                    "CineView 8K TV",
                    "SoundMax Soundbar",
                    "CineView OLED TV",
                ]
            ),
        },
    },
    {
        "customer_msg": """tell me about the CineView TV, the 8K one, Gamesphere console, the X one.
I'm on a budget, what computers do you have?""",
        "ideal_answer": {
            "Televisions and Home Theater Systems": set(["CineView 8K TV"]),
            "Gaming Consoles and Accessories": set(["GameSphere X"]),
            "Computers and Laptops": set(
                [
                    "TechPro Ultrabook",
                    "BlueWave Gaming Laptop",
                    "PowerLite Convertible",
                    "TechPro Desktop",
                    "BlueWave Chromebook",
                ]
            ),
        },
    },
    {
        "customer_msg": """What smartphones do you have?""",
        "ideal_answer": {
            "Smartphones and Accessories": set(
                [
                    "SmartX ProPhone",
                    "MobiTech PowerCase",
                    "SmartX MiniPhone",
                    "MobiTech Wireless Charger",
                    "SmartX EarBuds",
                ]
            )
        },
    },
    {
        "customer_msg": """I'm on a budget.  Can you recommend some smartphones to me?""",
        "ideal_answer": {
            "Smartphones and Accessories": set(
                [
                    "SmartX EarBuds",
                    "SmartX MiniPhone",
                    "MobiTech PowerCase",
                    "SmartX ProPhone",
                    "MobiTech Wireless Charger",
                ]
            )
        },
    },
    {
        "customer_msg": """What Gaming consoles would be good for my friend who is into racing games?""",
        "ideal_answer": {
            "Gaming Consoles and Accessories": set(
                [
                    "GameSphere X",
                    "ProGamer Controller",
                    "GameSphere Y",
                    "ProGamer Racing Wheel",
                    "GameSphere VR Headset",
                ]
            )
        },
    },
    {
        "customer_msg": """What could be a good present for my videographer friend?""",
        "ideal_answer": {
            "Cameras and Camcorders": set(
                [
                    "FotoSnap DSLR Camera",
                    "ActionCam 4K",
                    "FotoSnap Mirrorless Camera",
                    "ZoomMaster Camcorder",
                    "FotoSnap Instant Camera",
                ]
            )
        },
    },
    {
        "customer_msg": """I would like a hot tub time machine.""",
        "ideal_answer": [],
    },
]


if __name__ == "__main__":
    products_and_category = get_products_and_category()
    print("Product catalog loaded.\n")

    # Example queries with v1 prompt.
    print("=" * 50)
    print("v1 examples")
    for i, msg in enumerate(
        [
            """Which TV can I buy if I'm on a budget?""",
            """I need a charger for my smartphone""",
            """What computers do you have?""",
            """tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs do you have?""",
            """tell me about the CineView TV, the 8K one, Gamesphere console, the X one.
I'm on a budget, what computers do you have?""",
        ]
    ):
        print(f"\nExample {i}:")
        print(find_category_and_product_v1(msg, products_and_category))

    # Example queries with improved v2 prompt.
    print("\n" + "=" * 50)
    print("v2 examples")
    for i, msg in enumerate(
        [
            """tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs do you have?""",
            """Which TV can I buy if I'm on a budget?""",
        ]
    ):
        print(f"\nExample {i}:")
        print(find_category_and_product_v2(msg, products_and_category))

    # Automated evaluation against ideal answers.
    print("\n" + "=" * 50)
    print("Automated evaluation (v2)")
    score_accum = 0
    for i, pair in enumerate(msg_ideal_pairs_set):
        print(f"example {i}")
        customer_msg = pair["customer_msg"]
        ideal = pair["ideal_answer"]
        response = find_category_and_product_v2(
            customer_msg, products_and_category
        )
        score = eval_response_with_ideal(response, ideal, debug=False)
        print(f"{i}: {score}")
        score_accum += score

    n_examples = len(msg_ideal_pairs_set)
    fraction_correct = score_accum / n_examples
    print(f"Fraction correct out of {n_examples}: {fraction_correct}")
