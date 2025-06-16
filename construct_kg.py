# Loading necessary tools for dataset handling. LLM-based extraction, Neo4j graph writing and prompt formatting
import os
import hashlib
import json
import pathlib
import argparse
import ast

from dataset import get_dataset, get_dataset_items, extract_user_utterances
from models import LLM
from knowledge_graph import KnowledgeGraph
from prompts import *


def generate_schema_hash(schema, num_personas):
    """Generate a unique hash for the schema configuration"""
    schema_str = json.dumps(schema, sort_keys=True)
    return hashlib.md5(f"{schema_str}_{num_personas}".encode()).hexdigest()

def check_db_schema_match(kg, schema):
    """A helper function to compare current schema in the Neo4j database to the active schema configuration"""
    try:
        # Get current database schema info
        current_info = kg.get_current_schema_info()
        
        # If there's no data in the database, it's a fresh DB
        if not current_info["categories"]:
            return False
            
        # Extract schema categories and demographic fields into sets
        new_categories = set(config[0] for config in schema)
        new_demographic_fields = set()
        for config in schema:
            if config[0] == "demographics" and len(config) > 1 and config[1]:    # -- This means that it will only run if the name is demograohics, it includes second element (a list of fields) and that list is not empty
                new_demographic_fields = set(config[1])
        
        # Compare the currect KG schema (in Neo4j) with the new one
        categories_match = set(current_info["categories"]) == new_categories
        demographic_fields_match = set(current_info["demographic_fields"]) == new_demographic_fields
        
        return categories_match and demographic_fields_match
    except Exception as e:
        print(f"Error checking schema match: {str(e)}")
        return False

def load_knowledge_graph_from_file(filepath, neo4j_password):
    """A helper function to construct knowledge graph from a saved JSON file --> only called when --from-file argument is called from the command line
    
    Args:
        filepath: Path to the JSON file containing processed personas and schema
        neo4j_password: Password for the Neo4j database
    """
    print(f"Loading knowledge graph from file: {filepath}")
    
    # Load data from file
    try:
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        processed_personas = saved_data.get('processed_personas', {})
        schema = saved_data.get('schema', [])
        schema_hash = saved_data.get('schema_hash', '')
        
        print(f"Found {len(processed_personas)} personas with schema hash: {schema_hash}")
        
        # Initialize KnowledgeGraph with schema from file
        kg = KnowledgeGraph(uri="bolt://localhost:7690", user="neo4j", password=neo4j_password)
        
        # Drop existing database and create new one with proper schema
        print("Rebuilding database with schema from file...")
        kg.drop_database()
        kg.update_schema(schema, force_rebuild=False, skip_schema_check=True)
        
        # Process each persona from the file
        for i, (persona_id, persona_data) in enumerate(processed_personas.items(), 1):
            print(f"Processing persona {i}/{len(processed_personas)}: {persona_id[:8]}...")
            kg.upsert_persona(persona_data, persona_id)
        
        print(f"Successfully loaded {len(processed_personas)} personas into the knowledge graph")
        return True
    except Exception as e:
        print(f"Error loading knowledge graph from file: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Construct the knowledge graph from scratch or from a file')
    parser.add_argument('--from-file', type=str, help='Path to JSON file to load knowledge graph from')
    parser.add_argument('--list-files', action='store_true', help='List available JSON files in graphs directory')
    parser.add_argument('--similarity-threshold', type=float, default=0.75, help='Threshold for similarity matching')
    args = parser.parse_args()
        
    # Get Neo4j password
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    if not neo4j_password:
        print("Error: NEO4J_PKG_PASSWORD environment variable not set")
        return
    
    # If --list-files is specified, list available JSON files
    if args.list_files:
        results_dir = pathlib.Path("graphs")
        if not results_dir.exists():
            print("No graphs directory found")
            return
        
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print("No JSON files found in graphs directory")
            return
        
        print("Available JSON files:")
        for i, file in enumerate(json_files, 1):
            print(f"{i}. {file.name}")
        return
    
    # If --from-file is specified, construct knowledge graph from file
    if args.from_file:
        file_path = args.from_file
        # If the file path doesn't exist, check if it's in the graphs directory
        if not os.path.exists(file_path):
            file_path = os.path.join("graphs", file_path)
            if not os.path.exists(file_path):
                print(f"Error: File not found: {args.from_file}")
                return
        
        success = load_knowledge_graph_from_file(file_path, neo4j_password)
        if success:
            print("Knowledge graph construction from file completed successfully")
        else:
            print("Knowledge graph construction from file failed")
        return
    
    # Otherwise, proceed with normal construction from dataset
    # -------------------------------------------------------------------------
    # CUSTOMIZE YOUR SCHEMA HERE
    # -------------------------------------------------------------------------
    # Format: [category_name, fields_list]
    # - category_name: String name for the category
    # - fields_list: For demographics/basic categories, provide a list of fields
    #               For other categories, use None
    #
    # the Custom Schema
    #schema = [
    #    ["demographics", ["age", "gender", "ethnicity", "race", "nationality", "origin", "maritalStatus", 
    #    "familySize", "occupation", "employmentStatus", "educationLevel", "location", "sexuality", "religion", "income"]],
    #    ["pets", None],
    #    ["healthAndDisabilities", None],
    #    ["hobbiesAndInterests", None],
    #    ["favoritesAndPreferences", None],
    #    ["aspirationsAndGoals", None],
    #    ["personalityTraitsAndEmotions", None],
    #    ["familyRelationships", None],
    #    ["lifestyleAndHabits", None],
    #    ["additionalAttributes", None]
    #]
    
    # the PeaCoK Schema
    schema = [
        ["characteristics", None],
        ["routineOrHabit", None],
        ["aspirationsAndGoals", None],
        ["futureGoalOrPlan", None],
        ["familyRelationships", None],
        ["experience", None],
        ["relationship", None],
        ["addtionalAttributes", None]
    ]  
    # -------------------------------------------------------------------------
    
    # Set to False to use a subset of personas, True to use the entire dataset
    use_whole_dataset = False
    
    # Number of personas to process if not using the whole dataset
    num_personas = 3000
    
    # Generate a unique hash for the current schema
    if use_whole_dataset:
        num_personas = len(get_dataset_items(get_dataset(), "train", use_whole_dataset=True))
    schema_hash = generate_schema_hash(schema, num_personas)
    
    # Directory for saving results
    results_dir = pathlib.Path("graphs")
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / f"canonized_results_{schema_hash}.json"
    print(f"Using results file: {results_file}")
    
    # Determine if we need to force rebuild - default to True, but may change based on saved results
    force_rebuild = True
    
    # Previously processed personas (persona_id → canonized_result)
    processed_personas = {}
    
    # Default to 0 for the last processed index
    last_processed_index = 0
    
    # Check if we have saved results for the current schema and using whole dataset
    if results_file.exists():
        try:
            print(f"Found saved results for the current schema hash: {schema_hash}")
            with open(results_file, 'r') as f:
                saved_data = json.load(f)
                processed_personas = saved_data.get('processed_personas', {})
                last_processed_index = saved_data.get('last_processed_index', 0)
                
            print(f"Loaded {len(processed_personas)} previously processed personas")
            # Only set force_rebuild to False if we successfully loaded data
            force_rebuild = False
            
        except Exception as e:
            print(f"Error loading saved results: {str(e)}")
            print("Starting fresh with a forced rebuild")
            force_rebuild = True
            processed_personas = {}
            last_processed_index = 0
    else:
        print(f"No saved results found for schema hash: {schema_hash}")
        print("Starting fresh with a forced rebuild")


    # Connect to Neo4j and initialize KG with custom schema
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    
    # First initialize with just the connection, without providing schema yet
    kg = KnowledgeGraph(uri="bolt://localhost:7690", user="neo4j", password=neo4j_password)
    
    # Now manually handle database rebuild if needed instead of letting KG do it automatically
    if force_rebuild:
        print("Forcing database rebuild...")
        kg.drop_database()
    
    # After potential rebuild, update the schema but explicitly pass our force_rebuild flag
    # This will prevent the KnowledgeGraph from making its own decision to rebuild
    # Use the skip_schema_check=True when we've loaded valid saved results and don't want to force rebuild
    kg.update_schema(schema, force_rebuild=False, skip_schema_check=(not force_rebuild))
    
    # Initialize LLMs with prompts based on the custom schema
    persona_kg_extractor = LLM("GPT-4.1-mini", default_prompt=kg_prompt(schema=schema))
    # Initialize the canonicalizer without a default prompt since we'll specify it in each call
    persona_canonicalizer = LLM("GPT-4.1-mini")

    # Get dataset items (conversations with personas)
    dataset = get_dataset()
    dataset_items = get_dataset_items(dataset, "train", use_whole_dataset, num_personas)
    
    print(f"Using {'the entire dataset' if use_whole_dataset else 'a random sample'} of {len(dataset_items)} conversation items")
    
    # Display current schema information
    print("Using schema with these categories:")
    for category in schema:
        if len(category) > 1 and category[1]:
            print(f"- {category[0]} (with fields: {category[1]})")
        else:
            print(f"- {category[0]}")
    print()
    
    # Process each dataset item (conversation)
    item_count = len(dataset_items)
    for item_idx, item in enumerate(dataset_items[last_processed_index:], last_processed_index + 1):
        print(f"Processing dataset item {item_idx}/{item_count}...")
        
        # Process user 1 persona
        persona_user1 = item["user 1 personas"]
        persona_id_user1 = str(hashlib.sha256(persona_user1.encode('utf-8')).hexdigest())
        
        # Process user 2 persona
        persona_user2 = item["user 2 personas"]
        persona_id_user2 = str(hashlib.sha256(persona_user2.encode('utf-8')).hexdigest())
        
        # Process both personas sequentially
        for user_num, (persona, persona_id) in enumerate([(persona_user1, persona_id_user1), (persona_user2, persona_id_user2)], 1):
            # Skip if this persona was already processed
            if persona_id in processed_personas:
                print(f"Skipping already processed persona (User {user_num}): {persona_id[:8]}...")
                continue
                
            print(f"Processing persona (User {user_num}):\n{persona}...")
            
            # Extract user utterances from the dialogue
            dialogue = item["Best Generated Conversation"]
            utterances = extract_user_utterances(dialogue, user_num)
            
            # Generate structured attributes from the persona with the LLM using a pre-built prompt
            res = persona_kg_extractor.generate(prompt_params={"persona": persona}, json_output=True)
            # If the LLM returns a string, try to parse it into a dictionary
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except json.JSONDecodeError:
                    try:
                        res = ast.literal_eval(res)  # fallback for Python-style strings
                    except Exception as e:
                        print("Error parsing LLM response:", e)
                        continue  # Skip this persona if it's not parsable
            print("Extracted categories:", list(res.keys()))
            
            # Find similar attributes and exact matches for more efficient canonicalization
            similar_attributes, exact_match_attributes = kg.find_similar_attributes(res, threshold=args.similarity_threshold)
            try:
                # Calculate total number of similar attributes (non-exact matches)
                total_similar = 0
                for v in similar_attributes.values():
                    if isinstance(v, list):
                        total_similar += len(v)
                    elif isinstance(v, dict):
                        total_similar += sum(len(item) if isinstance(item, list) else 1 for item in v.values())
                
                # Calculate total number of exact matches already in the KG
                total_exact = 0
                for v in exact_match_attributes.values():
                    if isinstance(v, list):
                        total_exact += len(v)
                    elif isinstance(v, dict):
                        total_exact += sum(len(item) if isinstance(item, list) else 1 for item in v.values())
                
                # Report total counts of similar and exact match attributes
                if total_exact > 0:
                    print(f"Found {total_exact} exact matches in the knowledge graph")
                
                # Only perform canonicalization if we actually found similar (but not exact) attributes from the KG
                if total_similar > 0:
                    print(f"Found {total_similar} similar attributes that need canonicalization")
                    
                    # Get the canonicalization prompts
                    prompts_dict = canonicalization_prompt()
                    
                    # Run canonicalization with similar attributes and original attributes
                    canonized_res = persona_canonicalizer.generate(
                        prompt=prompts_dict,
                        prompt_params={"similar_attributes": similar_attributes, "persona_json": res}, 
                        json_output=True
                    )
                    
                    # Keep exact matches as they are
                    if total_exact > 0:
                        # Merge the canonized result with the exact matches
                        for category, values in exact_match_attributes.items():
                            if isinstance(values, list):
                                # For list-based categories
                                if category not in canonized_res:
                                    canonized_res[category] = values
                                else:
                                    # Add any missing items
                                    existing_items = canonized_res[category] if isinstance(canonized_res[category], list) else []
                                    canonized_res[category] = list(set(existing_items + values))
                            elif isinstance(values, dict):
                                # For field-based categories
                                if category not in canonized_res:
                                    canonized_res[category] = {}
                                for field, field_values in values.items():
                                    canonized_res[category][field] = field_values
                else:
                    # No similar attributes found, use the original result
                    canonized_res = res
                
                # If we received a string, we need to handle parsing it
                if isinstance(canonized_res, str):
                    print(f"Received string response, attempting to parse as JSON...")
                    try:
                        # First try standard JSON parsing
                        canonized_res = json.loads(canonized_res)
                    except json.JSONDecodeError as e:
                        print(f"Standard JSON parsing failed: {e}")
                        
                        # If it fails, try to fix Python dict syntax (single quotes to double quotes)
                        try:
                            # Use ast to safely evaluate the Python literal
                            python_dict = ast.literal_eval(canonized_res)
                            # Convert to JSON-compatible format
                            canonized_res = json.loads(json.dumps(python_dict))
                            print("Successfully converted Python dict to JSON")
                        except Exception as e2:
                            print(f"Cannot parse response as either JSON or Python dict: {e2}")
                            print(f"Skipping this persona due to parsing issues")
                            continue
                
                # Add utterances to the knowledge graph regardless of schema
                # This is done after canonicalization to ensure it's not affected by similarity checks
                if "utterances" not in canonized_res:
                    canonized_res["utterances"] = utterances
                
                # If we get here, we have a valid JSON object to save
                kg.upsert_persona(canonized_res, persona_id)
                
                processed_personas[persona_id] = canonized_res
                # Save progress every 5 items or when we're at the end
                if item_idx % 5 == 0 or item_idx == item_count:
                    print(f"Saving progress at item {item_idx}/{item_count}...")
                    try:
                        with open(results_file, 'w') as f:
                            json.dump({
                                'processed_personas': processed_personas,
                                'last_processed_index': item_idx - 1,
                                'schema_hash': schema_hash,
                                'schema': schema
                            }, f)
                        print(f"Saved progress to {results_file}")
                    except Exception as e:
                        print(f"Error saving progress: {str(e)}")
                            
            except Exception as e:
                print(f"Error processing persona: {str(e)}")
                print(f"Skipping persona and continuing...")
                continue
            print(f"Processed persona {persona_id[:8]}\n")
    
    print("All dataset items processed successfully!")
    
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'processed_personas': processed_personas,
                'last_processed_index': len(dataset_items) - 1,
                'schema_hash': schema_hash,
                'schema': schema,
                'completed': True
            }, f)
        print(f"Saved final results to {results_file}")
    except Exception as e:
        print(f"Error saving final results: {str(e)}")
    
    print(f"The knowledge graph now contains {len(processed_personas)} personas with the new schema and their utterances.")
    print("You can query the Neo4j database to explore the results.")


if __name__ == "__main__":
    main()