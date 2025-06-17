# Setting up standard libraries and third-party tools for data loading and processing
import os 
import json
import random

from datasets import load_dataset
from difflib import SequenceMatcher

# Using Hugging Face's 'dataset' library to download and load the dataset
def get_dataset():
    return load_dataset("google/Synthetic-Persona-Chat")

# Getting all or a random subset of conversation items from a dataset split
def get_dataset_items(dataset, split="train", use_whole_dataset=True, num_items=100):
    """
    Get conversation items (rows/entries) from the dataset split, either all or a random subset.
    
    Args:
        dataset: The loaded dataset
        split: The dataset split to use (train or test)
        use_whole_dataset: Whether to use the entire dataset or a subset
        num_items: Number of items to retrieve if not using the whole dataset
        
    Returns:
        List of dictionary items containing conversations and personas
    """
    # All items in the dataset
    all_items = dataset[split]
    
    if use_whole_dataset:
        # Convert to a list of dictionaries for consistent handling
        # This ensures we return the same format regardless of use_whole_dataset setting
        items_list = []
        for i in range(len(all_items)):
            items_list.append({
                "user 1 personas": all_items[i]["user 1 personas"],
                "user 2 personas": all_items[i]["user 2 personas"],
                "Best Generated Conversation": all_items[i]["Best Generated Conversation"]
            })
        return items_list
    else:
        # Get a deterministic sample with fixed seed
        random.seed(42)
        indices = random.sample(range(len(all_items)), min(num_items, len(all_items)))
        # Reset the random seed
        random.seed()
        
        # Create a list of dictionaries with just the selected indices
        selected_items = []
        for i in indices:
            selected_items.append({
                "user 1 personas": all_items[i]["user 1 personas"],
                "user 2 personas": all_items[i]["user 2 personas"],
                "Best Generated Conversation": all_items[i]["Best Generated Conversation"]
            })
        return selected_items

def extract_user_utterances(dialogue, user_number):
    """
    Extract utterances for a specific user from a conversation.
    
    Args:
        dialogue: The conversation dialogue
        user_number: The user number (1 or 2)
        
    Returns:
        String with the user's utterances, separated by newline characters
    """
    # Define the prefixes for each user in the dialogue
    user_prefix = f"User {user_number}:"
    
    utterances = []
    
    # Split the dialogue into lines and extract user's utterances
    lines = dialogue.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(user_prefix):
            # Remove the prefix and any leading/trailing whitespace
            utterance = line[len(user_prefix):].strip()
            utterances.append(utterance)
    
    # Join the utterances with newline characters
    return '\n'.join(utterances)


# Getting all the personas and using similarity functions to merge near-duplicate ones to reduce any redundancy in the KG:
"""
Utility functions for detecting and merging similar persona descriptions
    - similar() = checks the similairty between two strings
    - merge_sequences() = iterates through a list of personas and merges similar ones
    - merge_individual_sequences() = merges two similar persona strings line-by-line
"""

# Checks if two strings are similar based on their textual content using SequenceMatcher - 
def similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

# Merges similar persona strings from a list by comparing each pair using similar()
def merge_sequences(sequences, threshold=0.8):

    merged_sequences = []
    used_indices = set()

    for i, seq1 in enumerate(sequences):
        if i in used_indices:
            continue
        merged = False
        for j in range(i + 1, len(sequences)):
            if j not in used_indices and similar(seq1, sequences[j], threshold):
                merged_sequence = merge_individual_sequences(seq1, sequences[j])
                merged_sequences.append(merged_sequence)
                used_indices.update([i, j])
                merged = True
                break
        if not merged:
            merged_sequences.append(seq1)
            used_indices.add(i)

    return merged_sequences

# Merges two similar persona strings line-by-line, avoiding duplictae or highly similar lines
def merge_individual_sequences(seq1, seq2):

    sentences1 = seq1.split("\n")
    sentences2 = seq2.split("\n")

    merged_sentences = []
    used_indices = set()

    for s1 in sentences1:
        found_similar = False
        for idx, s2 in enumerate(sentences2):
            if idx not in used_indices and similar(s1, s2):
                merged_sentences.append(s1)
                used_indices.add(idx)
                found_similar = True
                break
        if not found_similar:
            merged_sentences.append(s1)

    for idx, s2 in enumerate(sentences2):
        if idx not in used_indices:
            merged_sentences.append(s2)

    return "\n".join(merged_sentences)

def get_personas(dataset, split="train", prune=True, threshold=0.6):
    """
    Legacy function to maintain compatibility with existing code.
    Gets all personas from the dataset, optionally merging similar ones.
    """
    merged_personas = dataset[split]["user 1 personas"] + dataset[split]["user 2 personas"]
    
    if prune:
        file_path = f"personas_pruned_{split}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)

        merged_personas = merge_sequences(merged_personas, threshold)
        with open(file_path, "w") as f:
            json.dump(merged_personas, f)

    return merged_personas

