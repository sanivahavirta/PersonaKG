def get_next_utterance_prompt(user1_persona, user2_persona, conversation_history, target_speaker, kg_info=""):
    """Generate a prompt for next utterance prediction with Chain-of-Thought prompting."""
    
    system_content = """You are tasked with predicting the next utterance in a conversation between two users with specific personas."""
    
    user_content = f"""Here are the personas:

User 1 Persona:
{user1_persona}

User 2 Persona:
{user2_persona}

Here is the conversation history so far:
{conversation_history}"""
    
    if kg_info:
        user_content += f"""

Additional information from the knowledge graph:
{kg_info}

Only use knowledge graph information if it directly continues the topic or emotional tone of the last message. Do not introduce unrelated hobbies, facts, or preferences — even if they are accurate.

Then, follow these rules:

- Begin by considering only the direct attributes of {target_speaker}.
- Use traits from similar personas only if they help explain or predict how {target_speaker} would behave in this specific conversation.
- If any example utterances from similar personas match the tone or topic of this dialogue, you may align the response to that style.
- Do not include information from the KG unless it directly supports the next likely reply based on the conversation history.
- If in doubt, prioritize the conversation flow over knowledge graph detail.
- Do not copy the style or topics of similar personas unless they exactly match the current dialogue.
"""

    # Chain-of-Thought instruction block (inserted before generation)
    user_content += f"""
Before you generate the next utterance, reflect step by step on the following:

1. What emotion, tone, or intent is expressed in the last message?
2. Which attributes or facts from the knowledge graph could personalize this reply, make it more engaging, or subtly reflect the speaker’s background?
3. How would a person with {target_speaker}'s attributes naturally respond in this situation?

Use this reasoning to decide on the response content and tone.

Now, predict what {target_speaker} would say next.

Your task is to generate ONLY the next utterance for the specified speaker using simple sentences. Leave out any additional commentary, explanation or supportive comments. Do not include the speaker's name in your response. 
If the previous message expresses enthusiasm or a personal interest, you may briefly mirror that tone. Additionally, if the previous message expresses an opinion or emotion, a short response is appropriate when it reflects the speaker’s alignment.
If no clear KG attribute applies, proceed naturally with the conversation. Prioritize conversation flow and keep responses short.

Next utterance:"""
    
    return [{
        "role": "system",
        "content": system_content
    },
    {
        "role": "user",
        "content": user_content
    }]


def canonicalization_prompt():
    """Generate a prompt for canonicalizing persona attributes.
    
    Returns:
        list: The prompt messages for the LLM.
    """
    system_content = """You are an expert assistant for canonicalizing persona attributes.
    Your task is to take in a set of attributes and canonicalize them.
    This means ensuring consistent naming and format across similar attributes.
    
    For example, if the existing knowledge graph has 'enjoys running' and 
    the new attribute is 'likes to run marathons', you should align the new
    attribute with the existing one, resulting in 'enjoys running' as the output.
    
    When working with demographics, normalize the values within each key.
    For example, 'Age: early 20s' and 'Age: 22' should both be standardized to 'Age: early 20s' or 'Age: 22'."""

    user_content_with_similars = """Please canonicalize the following persona attributes to match with existing attributes when appropriate.
    Follow these guidelines:
    1. If an attribute in the new persona is similar to one in the existing knowledge graph, use the existing version.
    2. For each category and attribute, check if it matches any similar existing ones before deciding to keep the new version.
    3. Do not add any explanatory text before or after the JSON.
    4. Output ONLY the clean JSON object with double quotes, nothing else.
    5. Do not use single quotes in your JSON output. Always use double quotes.
    
    Important: Failure to output a valid JSON with double quotes will cause system errors.
    Here are the similar existing attributes in our knowledge graph (threshold >= 0.95): {similar_attributes}
    Here's the new persona attributes: {persona_json}
    Please canonicalize the new persona attributes, ensuring they are in the cleanest, most normalized form possible."""
    
    return [{
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content_with_similars
        }]

def kg_prompt(schema=None):
    """Generate a prompt for extracting persona attributes based on a schema.

    Args:
        schema (list, optional): A custom schema for the knowledge graph.
            A list of lists where each inner list contains:
            [0] - category name (str)
            [1] - fields (list of str or None)

    Returns:
        list: The prompt messages for the LLM.
    """
    if not schema:
        # If no schema provided, create a very minimal default
        schema = [["basics", ["age", "occupation"]], ["traits", None], ["interests", None]]

    # Generate categories list from schema
    categories = [category_config[0] for category_config in schema]
    categories_text = "\n".join([f"- {cat}" for cat in categories])

    # Process field categories (like demographics/basics)
    field_categories = {}
    for category_config in schema:
        if len(category_config) > 1 and category_config[1]:
            cat_name = category_config[0]
            fields = category_config[1]
            field_categories[cat_name] = "\n".join([f"- \"{field}\"" for field in fields])

    # Generate field instructions section
    field_instructions = ""
    for cat_name, fields_text in field_categories.items():
        field_instructions += f"- For `{cat_name}`, include the following fields:\n{fields_text}\n"

    general_instructions = """- For all other categories, use arrays of strings. If the persona text doesn't mention anything relevant, use an empty array (`[]`)."""

    # Create the system content as a single non-formatted string to avoid issues
    system_content = "You are a persona attribute extraction system."

    user_content = f"""You will be given a persona's text.
        Your task is to extract the persona's attributes and return them in a strict JSON format, without any additional keys or text. 
        Follow these instructions exactly:

        1. **Schema**: You must produce valid JSON with the following top-level keys (using exactly these names):
{categories_text}

        2. **Field Instructions**:
{field_instructions}
{general_instructions}

        3. **Unique Assignment Rule**:
        - Each extracted value must appear in **only one** top-level category.
        - Choose the **most appropriate** category based on context.
        - **Do not duplicate** the same value across multiple categories.

        4. **Attribute Normalization**:
        - All attributes should be normalized to their most concise, standardized form.
        - For example, instead of "likes going to the gym", use "goes to gym".
        - Always come up with the most normalized form that captures the core concept.

        5. **JSON FORMAT REQUIREMENTS** (EXTREMELY IMPORTANT):
        - You MUST use double quotes (not single quotes) for ALL keys and string values.
        - Your output must be a valid JSON object that can be parsed by json.loads().
        - Do not use single quotes for keys or string values.
        - Do not add any text before or after the JSON object.
        - Your response should contain ONLY the properly formatted JSON."""

    user_content += "Here's the persona text: {persona}"

    return [{
            "role": "system",
            "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }]
