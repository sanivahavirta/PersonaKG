import os
import json
import argparse
import hashlib
import datetime
from tqdm import tqdm
from dataset import get_dataset, extract_user_utterances
from evaluate import load
import re
import random
import ast
from prompts import get_next_utterance_prompt, canonicalization_prompt, kg_prompt

from construct_kg import load_knowledge_graph_from_file
from models import LLM
from knowledge_graph import KnowledgeGraph

def setup_args():
    parser = argparse.ArgumentParser(description='Run next utterance prediction experiment')
    parser.add_argument('--model', "-m", type=str, default='GPT-4.1')
    parser.add_argument('--split', "-s", type=str, default='test', 
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use for evaluation')
    parser.add_argument('--knowledge_graph', "-kg", type=str, default=None,
                        help='Path to a knowledge graph file. If provided, will use the KG for persona information')
    parser.add_argument('--max_neighbors', "-mn", type=int, default=4,
                        help='Maximum number of neighboring personas to include in KG context')
    parser.add_argument('--output_dir', "-od", type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--similarity-threshold', type=float, default=0.75, help='Threshold for similarity matching')
    parser.add_argument('--num_samples', "-n", type=int, default=-1,
                        help='Number of random samples to use. -1 means use all samples (default: -1)')
    return parser.parse_args()

def parse_conversation(conversation):
    lines = conversation.strip().split('\n')
    utterances = []
    
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker = parts[0].strip()
            text = parts[1].strip()
            utterances.append({"speaker": speaker, "text": text})
    
    return utterances

def create_prediction_samples(data, split='test'):
    samples = []
    dataset_split = data[split]
    
    # Process all items sequentially
    for idx in range(len(dataset_split)):
        if dataset_split[idx]['Best Generated Conversation']:
            user1_persona = dataset_split[idx]['user 1 personas']
            user2_persona = dataset_split[idx]['user 2 personas']
            conversation = dataset_split[idx]['Best Generated Conversation']
            
            utterances = parse_conversation(conversation)
            
            for i in range(len(utterances)):
                history = utterances[:i]
                target = utterances[i]['text'] if i < len(utterances) else None
                
                if target is not None:
                    samples.append({
                        'user1_persona': user1_persona,
                        'user2_persona': user2_persona,
                        'history': history,
                        'target_speaker': utterances[i]['speaker'],
                        'target': target
                    })
    
    return samples

def predict_next_utterance(sample, llm, kg=None, kg_extractor=None, canonicalizer=None, max_neighbors=3, return_prompt=False):
    user1_persona = sample['user1_persona']
    user2_persona = sample['user2_persona']
    history = sample['history']
    target_speaker = sample['target_speaker']
    
    # Process personas through knowledge graph if provided
    kg_info = None
    if kg is not None and kg_extractor is not None and canonicalizer is not None:
        # Generate unique IDs for personas
        user1_id = str(hashlib.sha256(user1_persona.encode('utf-8')).hexdigest())
        user2_id = str(hashlib.sha256(user2_persona.encode('utf-8')).hexdigest())
        
        # Extract conversation as a formatted string for each user
        conversation_text = '\n'.join([f"User {u['speaker'].split()[-1]}: {u['text']}" for u in history])
        user1_utterances = extract_user_utterances(conversation_text, 1) if conversation_text else ""
        user2_utterances = extract_user_utterances(conversation_text, 2) if conversation_text else ""
        
        with kg.driver.session() as session:
            # Check if personas already exist
            user1_exists = session.run("MATCH (p:Persona {id: $id}) RETURN p", id=user1_id).single()
            user2_exists = session.run("MATCH (p:Persona {id: $id}) RETURN p", id=user2_id).single()
            
            # Process user1 persona if it doesn't exist
            if not user1_exists:
                try:
                    print(f"Adding User 1 persona to knowledge graph...")
                    # Extract attributes using LLM
                    extracted_attrs = kg_extractor.generate(prompt_params={"persona": user1_persona}, json_output=True)
                    
                    # Find similar attributes and exact matches for more efficient canonicalization
                    similar_attrs, exact_match_attrs = kg.find_similar_attributes(extracted_attrs, threshold=args.similarity_threshold)
                    
                    # Calculate total number of similar attributes (non-exact matches)
                    total_similar = 0
                    for v in similar_attrs.values():
                        if isinstance(v, list):
                            total_similar += len(v)
                        elif isinstance(v, dict):
                            total_similar += sum(len(item) if isinstance(item, list) else 1 for item in v.values())
                    
                    # Calculate total number of exact matches
                    total_exact = 0
                    for v in exact_match_attrs.values():
                        if isinstance(v, list):
                            total_exact += len(v)
                        elif isinstance(v, dict):
                            total_exact += sum(len(item) if isinstance(item, list) else 1 for item in v.values())
                    
                    # Report total counts of similar and exact match attributes
                    if total_exact > 0:
                        print(f"Found {total_exact} exact matches in the knowledge graph for User 1")
                    
                    canonized_attrs = extracted_attrs
                    
                    # Only perform canonicalization if we actually found similar (but not exact) attributes
                    if total_similar > 0:
                        print(f"Found {total_similar} similar attributes that need canonicalization for User 1")
                        
                        # Get the canonicalization prompts
                        prompts_dict = canonicalization_prompt()
                        
                        # Run canonicalization with similar attributes
                        canonized_attrs = canonicalizer.generate(
                            prompt=prompts_dict,
                            prompt_params={"similar_attributes": similar_attrs, "persona_json": extracted_attrs},
                            json_output=True
                        )
                        
                        # Keep exact matches as they are
                        if total_exact > 0:
                            # Merge the canonized result with the exact matches
                            for category, values in exact_match_attrs.items():
                                if isinstance(values, list):
                                    # For list-based categories
                                    if category not in canonized_attrs:
                                        canonized_attrs[category] = values
                                    else:
                                        # Add any missing items
                                        existing_items = canonized_attrs[category] if isinstance(canonized_attrs[category], list) else []
                                        canonized_attrs[category] = list(set(existing_items + values))
                                elif isinstance(values, dict):
                                    # For field-based categories
                                    if category not in canonized_attrs:
                                        canonized_attrs[category] = {}
                                    for field, field_values in values.items():
                                        canonized_attrs[category][field] = field_values
                    
                    # Handle string response if needed
                    if isinstance(canonized_attrs, str):
                        try:
                            canonized_attrs = json.loads(canonized_attrs)
                        except json.JSONDecodeError:
                            try:
                                python_dict = ast.literal_eval(canonized_attrs)
                                canonized_attrs = json.loads(json.dumps(python_dict))
                            except Exception:
                                print("Error parsing canonized attributes")
                                canonized_attrs = extracted_attrs
                    
                    # Add user1 utterances to the canonized attributes
                    if user1_utterances:
                        canonized_attrs["utterances"] = user1_utterances
                    
                    # Add to knowledge graph directly - the KG already has the right schema
                    kg.upsert_persona(canonized_attrs, user1_id)
                except Exception as e:
                    print(f"Error adding User 1 persona to knowledge graph: {str(e)}")
            
            # Process user2 persona if it doesn't exist
            if not user2_exists:
                try:
                    print(f"Adding User 2 persona to knowledge graph...")
                    # Extract attributes using LLM
                    extracted_attrs = kg_extractor.generate(prompt_params={"persona": user2_persona}, json_output=True)
                    
                    # Find similar attributes and exact matches for more efficient canonicalization
                    similar_attrs, exact_match_attrs = kg.find_similar_attributes(extracted_attrs, threshold=args.similarity_threshold)
                    
                    # Calculate total number of similar attributes (non-exact matches)
                    total_similar = 0
                    for v in similar_attrs.values():
                        if isinstance(v, list):
                            total_similar += len(v)
                        elif isinstance(v, dict):
                            total_similar += sum(len(item) if isinstance(item, list) else 1 for item in v.values())
                    
                    # Calculate total number of exact matches
                    total_exact = 0
                    for v in exact_match_attrs.values():
                        if isinstance(v, list):
                            total_exact += len(v)
                        elif isinstance(v, dict):
                            total_exact += sum(len(item) if isinstance(item, list) else 1 for item in v.values())
                    
                    # Report total counts of similar and exact match attributes
                    if total_exact > 0:
                        print(f"Found {total_exact} exact matches in the knowledge graph for User 2")
                    
                    canonized_attrs = extracted_attrs
                    
                    # Only perform canonicalization if we actually found similar (but not exact) attributes
                    if total_similar > 0:
                        print(f"Found {total_similar} similar attributes that need canonicalization for User 2")
                        
                        # Get the canonicalization prompts
                        prompts_dict = canonicalization_prompt()
                        
                        # Run canonicalization with similar attributes
                        canonized_attrs = canonicalizer.generate(
                            prompt=prompts_dict,
                            prompt_params={"similar_attributes": similar_attrs, "persona_json": extracted_attrs},
                            json_output=True
                        )
                        
                        # Keep exact matches as they are
                        if total_exact > 0:
                            # Merge the canonized result with the exact matches
                            for category, values in exact_match_attrs.items():
                                if isinstance(values, list):
                                    # For list-based categories
                                    if category not in canonized_attrs:
                                        canonized_attrs[category] = values
                                    else:
                                        # Add any missing items
                                        existing_items = canonized_attrs[category] if isinstance(canonized_attrs[category], list) else []
                                        canonized_attrs[category] = list(set(existing_items + values))
                                elif isinstance(values, dict):
                                    # For field-based categories
                                    if category not in canonized_attrs:
                                        canonized_attrs[category] = {}
                                    for field, field_values in values.items():
                                        canonized_attrs[category][field] = field_values
                    
                    # Handle string response if needed
                    if isinstance(canonized_attrs, str):
                        try:
                            canonized_attrs = json.loads(canonized_attrs)
                        except json.JSONDecodeError:
                            try:
                                python_dict = ast.literal_eval(canonized_attrs)
                                canonized_attrs = json.loads(json.dumps(python_dict))
                            except Exception:
                                print("Error parsing canonized attributes")
                                canonized_attrs = extracted_attrs
                    
                    # Add user2 utterances to the canonized attributes
                    if user2_utterances:
                        canonized_attrs["utterances"] = user2_utterances
                    
                    # Add to knowledge graph directly - the KG already has the right schema
                    kg.upsert_persona(canonized_attrs, user2_id)
                except Exception as e:
                    print(f"Error adding User 2 persona to knowledge graph: {str(e)}")
            
        # Get KG information for the target persona and neighbors
        kg_info = ""
        target_id = user1_id if target_speaker == "User 1" else user2_id
        
        with kg.driver.session() as session:
            # 1. First get direct attributes of the target persona
            kg_info = f"Knowledge Graph Context for {target_speaker}:\n\n"
            kg_info += f"== {target_speaker}'s Attributes ==\n"
            
            attributes_query = """
            MATCH (p:Persona {id: $id})-[r]->(a:Attribute)
            RETURN a.category AS category, a.key AS key, a.value AS value
            ORDER BY a.category, a.key
            """
            
            attributes = session.run(attributes_query, id=target_id)
            
            # Format KG information by category
            category_data = {}
            
            # Group by category first
            for record in attributes:
                category = record["category"]
                key = record["key"]
                value = record["value"]
                
                if category not in category_data:
                    category_data[category] = []
                
                if key:
                    category_data[category].append(f"{key}: {value}")
                else:
                    category_data[category].append(value)
            
            # Format by category
            for category, values in category_data.items():
                kg_info += f"\n{category.capitalize()}:\n"
                for value in values:
                    kg_info += f"- {value}\n"
            
            # 2. Find neighbors with shared attributes            
            neighbors_query = """
            MATCH (p1:Persona {id: $id})-[r1]->(a:Attribute)<-[r2]-(p2:Persona)
            WHERE p1 <> p2
            WITH p2, count(a) AS shared_count, collect(a.value) AS shared_values
            ORDER BY shared_count DESC
            LIMIT $max_neighbors
            RETURN p2.id AS neighbor_id, shared_count, shared_values
            """
            
            neighbors = list(session.run(neighbors_query, id=target_id, max_neighbors=max_neighbors))
            
            if neighbors:
                kg_info += f"\n== Personas with Similar Attributes ==\n"
                
            for neighbor in neighbors:
                neighbor_id = neighbor["neighbor_id"]
                shared_count = neighbor["shared_count"]
                shared_values = neighbor["shared_values"]
                
                kg_info += f"\nNeighbor {neighbor_id} shares {shared_count} attributes:\n"
                for value in shared_values:
                    kg_info += f"- {value}\n"
                
                neighbor_info_query = """
                MATCH (p:Persona {id: $id})-[r]->(a:Attribute)
                RETURN a.category AS category, a.key AS key, a.value AS value, type(r) AS relationship
                ORDER BY a.category, a.key
                """
                
                neighbor_attributes = session.run(neighbor_info_query, id=neighbor_id)
                
                categorized_attributes = {}
                for record in neighbor_attributes:
                    category = record["category"]
                    value = record["value"]
                    rel = record["relationship"]
                    
                    if category not in categorized_attributes:
                        categorized_attributes[category] = []
                    
                    if value in shared_values:
                        continue
                        
                    categorized_attributes[category].append((value, rel))
                
                if categorized_attributes:
                    kg_info += "Other attributes:\n"
                    for category, attribute_list in categorized_attributes.items():
                        if attribute_list:
                            kg_info += f"{category.capitalize()}:\n"
                            for value, rel in attribute_list:
                                kg_info += f"- {value} ({rel})\n"
                
                utterances_query = """
                MATCH (p:Persona {id: $id})
                RETURN p.utterances AS utterances
                """
                
                utterances_result = session.run(utterances_query, id=neighbor_id).single()
                if utterances_result and utterances_result["utterances"]:
                    utterances = utterances_result["utterances"]
                    kg_info += "Utterances:\n"
                    utterance_lines = utterances.split("\n")
                    for utterance in utterance_lines:
                        kg_info += f"- \"{utterance}\"\n"
    
    formatted_history = ""
    for utterance in history:
        formatted_history += f"{utterance['speaker']}: {utterance['text']}\n"
    
    prompt = get_next_utterance_prompt(
        user1_persona=user1_persona,
        user2_persona=user2_persona,
        conversation_history=formatted_history,
        target_speaker=target_speaker,
        kg_info=kg_info
    )
    
    prediction = llm.generate(prompt)
    prediction = re.sub(r'^.*?:', '', prediction).strip()
    
    if return_prompt:
        return prediction, prompt
    else:
        return prediction

def evaluate_predictions(predictions, targets):
    if not predictions or not targets:
        return {'bleu': 0, 'rouge': {'precision': 0, 'recall': 0, 'f1': 0}}
    
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
    
    references = [[t] for t in targets]
    
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    bleu_score = bleu_result["bleu"]

    rouge_result = rouge_metric.compute(predictions=predictions, references=targets)
    
    return {
        'bleu': bleu_score,
        'rouge': rouge_result
    }

def create_experiment_id(args):
    """Create a unique identifier for an experiment based on its parameters"""
    params = {
        'model': args.model,
        'split': args.split,
        'knowledge_graph': args.knowledge_graph,
        'max_neighbors': args.max_neighbors,
        'num_samples': args.num_samples
    }
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def run_experiment(args):
    data = get_dataset()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a unique experiment ID based on parameters
    experiment_id = create_experiment_id(args)
    checkpoint_file = os.path.join(args.output_dir, f"checkpoint_{experiment_id}.json")
    eval_file = os.path.join(args.output_dir, f"eval_{experiment_id}.json")
    final_output_file = os.path.join(args.output_dir, f"results_{experiment_id}.json")
    
    # Check if this experiment has already been completed
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            results_data = json.load(f)
    
    # Check if there's a checkpoint to resume from
    completed_samples = []
    predictions = []
    targets = []
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file. Resuming experiment from checkpoint.")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            completed_samples = checkpoint_data.get('completed_samples', [])
            predictions = checkpoint_data.get('predictions', [])
            targets = checkpoint_data.get('targets', [])
            start_idx = len(completed_samples)
        print(f"Resuming from sample {start_idx}")
    else:
        print(f"Starting new experiment with ID: {experiment_id}")
    
    # Initialize knowledge graph if specified
    kg = None
    schema = None
    kg_extractor = None
    canonicalizer = None
    
    if args.knowledge_graph:
        print(f"Loading knowledge graph from file: {args.knowledge_graph}")
        neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
        if not neo4j_password:
            print("Error: NEO4J_PKG_PASSWORD environment variable not set. Knowledge graph will not be used.")
        else:
            # Determine the file path
            if not os.path.exists(args.knowledge_graph):
                # Try prefixing with graphs/
                alt_path = os.path.join("graphs", args.knowledge_graph)
                if os.path.exists(alt_path):
                    args.knowledge_graph = alt_path
                else:
                    print(f"Knowledge graph file not found: {args.knowledge_graph}")
                    return
                    
            # Use the load_knowledge_graph_from_file function
            print(f"Using load_knowledge_graph_from_file with {args.knowledge_graph}")
            try:
                # Load schema from the file
                with open(args.knowledge_graph, 'r') as f:
                    saved_data = json.load(f)
                
                schema = saved_data.get('schema', [])
                if not schema:
                    print("No schema found in the file. Cannot proceed.")
                    return
                
                # Create a temporary KG object to check the current schema
                temp_kg = KnowledgeGraph(uri="bolt://localhost:7690", user="neo4j", password=neo4j_password)
                
                # Check if the schema in the file matches the existing KG
                rebuild_needed = False
                try:
                    # First check if there's any data in the KG
                    with temp_kg.driver.session() as session:
                        result = session.run("MATCH (n) RETURN count(n) as count").single()
                        if result and result["count"] > 0:
                            # There's data, check if schema matches
                            print("Found existing knowledge graph. Checking schema compatibility...")
                            if temp_kg.schema_requires_rebuild(schema):
                                print("Schema mismatch detected. Rebuilding knowledge graph...")
                                rebuild_needed = True
                            else:
                                print("Existing schema is compatible. Using existing knowledge graph.")
                        else:
                            # Empty database, needs rebuild
                            print("Knowledge graph is empty. Building from file...")
                            rebuild_needed = True
                except Exception as e:
                    print(f"Error checking schema: {str(e)}")
                    rebuild_needed = True
                
                # If rebuild is needed, use load_knowledge_graph_from_file
                if rebuild_needed:
                    print(f"Loading knowledge graph from file: {args.knowledge_graph}")
                    success = load_knowledge_graph_from_file(args.knowledge_graph, neo4j_password)
                    if not success:
                        print("Failed to load knowledge graph from file")
                        return
                
                # Now create our own KG object to use for queries
                # The database already has the correct schema from load_knowledge_graph_from_file
                kg = KnowledgeGraph(uri="bolt://localhost:7690", user="neo4j", password=neo4j_password)
                
                # Load schema from the file for our LLMs
                with open(args.knowledge_graph, 'r') as f:
                    saved_data = json.load(f)
                
                schema = saved_data.get('schema', [])
                if not schema:
                    print("No schema found in the file. Cannot proceed.")
                    return
                
                # Make sure our KG instance has the same schema
                kg.update_schema(schema, force_rebuild=False, skip_schema_check=True)
                
                # Display schema information
                print("Using schema with these categories:")
                for category in schema:
                    if len(category) > 1 and category[1]:
                        print(f"- {category[0]} (with fields: {category[1]})")
                    else:
                        print(f"- {category[0]}")
                
                # Initialize LLMs for persona extraction and canonicalization
                kg_extractor = LLM("GPT-4.1-mini", default_prompt=kg_prompt(schema=schema))
                canonicalizer = LLM("GPT-4.1-mini")
                
                print("Knowledge graph initialized with schema")
            except Exception as e:
                print(f"Error initializing knowledge graph: {str(e)}")
                kg = None
    
    print(f"Creating prediction samples from {args.split} split...")
    samples = create_prediction_samples(
        data, 
        split=args.split
    )
    
    # If num_samples is specified and valid, select a random subset
    if args.num_samples > 0 and args.num_samples < len(samples):
        random.seed(42)  # For reproducibility
        samples = random.sample(samples, args.num_samples)
        print(f"Randomly selected {len(samples)} samples.")
    
    # If we've already processed some samples, skip those
    if start_idx > 0:
        print(f"Skipping {start_idx} already processed samples")
        samples_to_process = samples[start_idx:]
    else:
        samples_to_process = samples
    
    if not samples_to_process:
        print("All samples have been processed. Proceeding to evaluation.")
    else:
        print(f"Running predictions with model {args.model} on {len(samples_to_process)} samples...")
        llm = LLM(args.model, gen_params={
            "temperature": 0.7,
            "max_tokens": 128
        })
    
    # Only process samples that haven't been processed yet
    for i, sample in enumerate(tqdm(samples_to_process)):
    
        prediction, prompt = predict_next_utterance(
            sample, 
            llm=llm,
            kg=kg,
            kg_extractor=kg_extractor,
            canonicalizer=canonicalizer,
            max_neighbors=args.max_neighbors,
            return_prompt=True
        )
        
        predictions.append(prediction)
        targets.append(sample['target'])
        
        # Record this sample as completed
        completed_samples.append({
            'index': start_idx + i,
            'user1_persona': sample['user1_persona'],
            'user2_persona': sample['user2_persona'],
            'history': sample['history'],
            'target': sample['target'],
            'prediction': prediction,
            'prompt': prompt
        })
        
        # Save checkpoint every 5 samples
        if (i + 1) % 5 == 0 or i == len(samples_to_process) - 1:
            checkpoint_data = {
                'args': vars(args),
                'experiment_id': experiment_id,
                'completed_samples': completed_samples,
                'timestamp': str(datetime.datetime.now()),
                'progress': f"{len(completed_samples)}/{len(samples)}"
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"\nCheckpoint saved at sample {start_idx + i + 1}/{len(samples)}")
        
        # Run intermediate evaluation every 10 samples
        if (start_idx + i + 1) % 10 == 0 or i == len(samples_to_process) - 1:
            print(f"\nRunning intermediate evaluation at sample {start_idx + i + 1}...")
            # Get predictions and targets from completed samples
            interim_predictions = [s['prediction'] for s in completed_samples]
            interim_targets = [s['target'] for s in completed_samples]
            
            # Run evaluation
            interim_results = evaluate_predictions(interim_predictions, interim_targets)
            
            # Create interim result data
            interim_result_data = {
                'args': vars(args),
                'experiment_id': experiment_id,
                'metrics': interim_results,
                'timestamp': str(datetime.datetime.now()),
                'total_samples': len(samples),
                'processed_samples': len(completed_samples),
            }
            
            # Save to a single evaluation file that gets updated each time
            with open(eval_file, 'w') as f:
                json.dump(interim_result_data, f, indent=2)                    
            print(f"Evaluation results updated in {eval_file}")
            print(f"Interim BLEU Score: {interim_results['bleu']}")
            print(f"Interim ROUGE-F1 Score: {interim_results['rouge']}")
 
    
    print("Evaluating predictions...")
    # Combine predictions and targets from both checkpoint and newly processed samples
    all_predictions = []
    all_targets = []
    
    # Add predictions and targets from completed samples (includes checkpoint data)
    for sample in completed_samples:
        all_predictions.append(sample['prediction'])
        all_targets.append(sample['target'])
    
    # Add any remaining predictions/targets that might not be in completed_samples yet
    for i in range(len(predictions)):
        if i >= len(completed_samples):
            all_predictions.append(predictions[i])
            all_targets.append(targets[i])
    
    # Run evaluation on all samples
    results = evaluate_predictions(all_predictions, all_targets)
    
    # Update the evaluation file with final results
    final_result_data = {
        'args': vars(args),
        'experiment_id': experiment_id,
        'metrics': results,
        'timestamp': str(datetime.datetime.now()),
        'total_samples': len(samples),
        'processed_samples': len(all_predictions)
    }
    
    with open(eval_file, 'w') as f:
        json.dump(final_result_data, f, indent=2)
    
    print(f"Final results saved to {eval_file}")
    print(f"BLEU Score: {results['bleu']}")
    print(f"ROUGE-F1 Score: {results['rouge']}")
    
    return results

if __name__ == "__main__":
    args = setup_args()
    run_experiment(args)