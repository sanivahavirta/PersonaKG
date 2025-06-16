from neo4j import GraphDatabase
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeGraph:
    def __init__(self, uri, user, password, schema=None):
        # Initialize the Neo4j driver.
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Default schema configuration if none provided
        self._set_default_schema()
        
        # Update schema if provided
        if schema:
            self.update_schema(schema)
    
    def _set_default_schema(self):
        """Set the default schema with predefined categories and fields."""
        # Define the allowed top-level categories in the JSON.
        self.allowed_categories = {
            "demographics",
            "pets",
            "healthAndDisabilities",
            "hobbiesAndInterests",
            "favoritesAndPreferences",
            "aspirationsAndGoals",
            "personalityTraitsAndEmotions",
            "familyRelationships",
            "lifestyleAndHabits",
            "additionalAttributes"
        }
        
        # Initialize field categories dictionary
        self.field_categories = {}
        
        # Define the allowed fields for the demographics category
        self.field_categories["demographics"] = {
            "age",
            "gender",
            "ethnicity",
            "race",
            "nationality",
            "origin",
            "maritalStatus",
            "familySize",
            "occupation",
            "employmentStatus",
            "educationLevel",
            "location",
            "sexuality",
            "religion",
            "income"
        }

        self.relationship_map = {
            "demographics": "HAS_DEMOGRAPHIC",
            "pets": "HAS_PET",
            "healthAndDisabilities": "HAS_HEALTH_ISSUES",
            "hobbiesAndInterests": "HAS_INTEREST",
            "favoritesAndPreferences": "HAS_PREFERENCE",
            "aspirationsAndGoals": "HAS_GOAL",
            "personalityTraitsAndEmotions": "HAS_PERSONALITY_TRAIT",
            "familyRelationships": "HAS_FAMILY_RELATIONSHIP",
            "lifestyleAndHabits": "HAS_LIFESTYLE",
            "additionalAttributes": "HAS_ADDITIONAL_ATTRIBUTE"
        }

    def get_current_schema_info(self):
        """Get information about the current schema from the database.
        
        Returns:
            dict: Information about the current schema, including:
                  - categories: list of category names from Attribute nodes
                  - relationship_types: list of relationship types used
                  - demographic_fields: list of demographic field keys
        """
        with self.driver.session() as session:
            # Get all categories from Attribute nodes
            categories_query = """
            MATCH (a:Attribute)
            RETURN DISTINCT a.category AS category
            """
            categories = [record["category"] for record in session.run(categories_query)]
            
            # Get all relationship types used in the graph
            rel_types_query = """
            MATCH ()-[r]->() 
            RETURN DISTINCT type(r) AS rel_type
            """
            rel_types = [record["rel_type"] for record in session.run(rel_types_query)]
            
            # Get all demographic field keys
            demographics_fields_query = """
            MATCH (a:Attribute {category: 'demographics'})
            WHERE a.key IS NOT NULL
            RETURN DISTINCT a.key AS field
            """
            demographic_fields = [record["field"] for record in session.run(demographics_fields_query)]
        
        return {
            "categories": categories,
            "relationship_types": rel_types,
            "demographic_fields": demographic_fields
        }
    
    def schema_requires_rebuild(self, new_schema):
        """Determine if the new schema is significantly different from the current schema.
        
        Args:
            new_schema (list): The new schema configuration
            
        Returns:
            bool: True if the schema change requires a database rebuild
        """
        # Get current schema info
        current_info = self.get_current_schema_info()
        
        # Extract new schema categories and demographic fields
        new_categories = set(config[0] for config in new_schema)
        new_demographic_fields = set()
        for config in new_schema:
            if config[0] == "demographics" and len(config) > 1 and config[1]:
                new_demographic_fields = set(config[1])
        
        # Check for significant changes
        categories_changed = current_info["categories"] and new_categories.difference(current_info["categories"])
        demographic_fields_changed = current_info["demographic_fields"] and new_demographic_fields.difference(current_info["demographic_fields"])
        
        return categories_changed or demographic_fields_changed
    
    def drop_database(self):
        """Drop all nodes and relationships in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database has been cleared.")
    
    def update_schema(self, schema, force_rebuild=False, skip_schema_check=False):
        """Update the knowledge graph schema with a new configuration.
        
        Args:
            schema (list): List of lists where each inner list contains:
                           [0] - category name (str)
                           [1] - fields (list of str or None if not demographics)
            force_rebuild (bool): If True, drop and recreate the database
            skip_schema_check (bool): If True, skip the schema change detection
        """
        # Check if we need to rebuild the database
        if force_rebuild:
            print("Forcing database rebuild due to explicit request...")
            self.drop_database()
        elif not skip_schema_check and self.schema_requires_rebuild(schema):
            print("Schema change detected. Dropping and recreating the database...")
            self.drop_database()
        
        # Clear existing schema configuration
        self.allowed_categories = set()
        self.field_categories = {}  # Store field-based categories and their fields
        self.relationship_map = {}
        
        # Process the new schema
        for category_config in schema:
            category = category_config[0]
            fields = category_config[1] if len(category_config) > 1 else None
            
            # Add to allowed categories
            self.allowed_categories.add(category)
            
            # Create relationship type (convert to uppercase with HAS_ prefix)
            rel_type = f"HAS_{category.upper()}"
            if category == "demographics":
                rel_type = "HAS_DEMOGRAPHIC"
            self.relationship_map[category] = rel_type
            
            # Store field-based categories and their fields
            if fields:
                self.field_categories[category] = set(fields)

    def close(self):
        self.driver.close()

    def validate_json(self, persona_json):
        """
        Validates that the input JSON only contains allowed categories,
        and for field-based categories (like basics/demographics/profile), only allowed fields.
        Special case: 'utterances' is always allowed regardless of schema.
        """
        # Check that all top-level keys are allowed, with 'utterances' as a special exception
        for key in persona_json:
            if key != 'utterances' and key not in self.allowed_categories:
                raise ValueError(
                    f"Unrecognized category in JSON: '{key}'. "
                    f"Allowed categories: {self.allowed_categories}"
                )
        
        # Validate all field-based categories (previously just demographics)
        for category, allowed_fields in getattr(self, 'field_categories', {}).items():
            if category in persona_json:
                category_data = persona_json.get(category, {})
                if isinstance(category_data, dict):
                    for field in category_data:
                        if field not in allowed_fields:
                            raise ValueError(
                                f"Unrecognized field '{field}' in '{category}' category. "
                                f"Allowed fields: {allowed_fields}"
                            )

    def compute_attribute_id(self, category, value, key=None):
        """
        Compute a unique id for an attribute based on its category, and value.
        For demographics, a key is also included.
        """
        if key:
            identifier = f"{category}_{key}_{value}"
        else:
            identifier = f"{category}_{value}"
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

    def get_existing_attributes(self):
        """
        Fetches all existing Attribute nodes in the graph, grouped by category.
        Returns a dict like: { "interestsAndHobbies": ["running", "reading"], ... }
        """
        query = """
        MATCH (a:Attribute)
        RETURN a.category AS category, a.value AS value
        ORDER BY a.category, a.value
        """
        existing = {}
        with self.driver.session() as session:
            results = session.run(query)
            for record in results:
                cat = record["category"]
                val = record["value"]
                if cat not in existing:
                    existing[cat] = []
                existing[cat].append(val)
        return existing
        
    def find_similar_attributes(self, persona_json, threshold=0.95):
        """
        Find existing attributes in the graph that are similar to the new persona attributes.
        
        Args:
            persona_json (dict): The new persona attributes
            threshold (float): Similarity threshold (0.0 to 1.0)
            
        Returns:
            tuple: (similar_attributes, exact_match_attributes, needs_canonicalization)
                - similar_attributes: A dictionary with similar attributes grouped by category
                - exact_match_attributes: A dictionary with exact match attributes (similarity = 1.0)
                - needs_canonicalization: Boolean indicating if any attributes need canonicalization
        """
        # Get all existing attributes
        all_existing = self.get_existing_attributes()
        
        # Initialize result dictionaries for similar and exact match attributes
        similar_attributes = {}    # For attributes that are similar but not exact
        exact_match_attributes = {}  # For attributes that are exact matches
        
        # For each category in the persona_json
        for category, values in persona_json.items():
            # Skip if category doesn't exist in the database
            if category not in all_existing:
                continue
                
            existing_values = all_existing[category]
            
            # Skip if no existing values
            if not existing_values:
                continue
                
            # Handle field-based categories (like demographics)
            if isinstance(values, dict):
                # Initialize category dictionaries if they don't exist
                if category not in similar_attributes:
                    similar_attributes[category] = {}
                if category not in exact_match_attributes:
                    exact_match_attributes[category] = {}
                    
                for field, field_value in values.items():
                    # Handle both single values and lists
                    field_values = field_value if isinstance(field_value, list) else [field_value]
                    
                    # Filter out None and empty values
                    field_values = [v for v in field_values if v]
                    
                    # Skip if no valid values
                    if not field_values:
                        continue
                    
                    # Find similar existing values and identify exact matches
                    similar_vals, exact_vals = self._compute_similarity_with_exact_match(
                        field_values, existing_values, threshold)
                    
                    # Store similar (non-exact) values if any
                    if similar_vals:
                        similar_attributes[category][field] = similar_vals
                        
                    # Store exact match values if any
                    if exact_vals:
                        exact_match_attributes[category][field] = exact_vals
                        
            # Handle list-based categories
            elif isinstance(values, list):
                # Filter out empty values
                values = [v for v in values if v]
                
                # Skip if no valid values
                if not values:
                    continue
                
                # Find similar existing values and identify exact matches
                similar_vals, exact_vals = self._compute_similarity_with_exact_match(
                    values, existing_values, threshold)
                
                # Store similar (non-exact) values if any
                if similar_vals:
                    similar_attributes[category] = similar_vals
                    
                # Store exact match values if any
                if exact_vals:
                    exact_match_attributes[category] = exact_vals
        
        return similar_attributes, exact_match_attributes
    
    def _compute_similarity(self, new_values, existing_values, threshold, return_exact_match=False):
        """
        Compute similarity between new values and existing values using cosine similarity.
        
        Args:
            new_values (list): List of new values
            existing_values (list): List of existing values
            threshold (float): Similarity threshold
            return_exact_match (bool): Whether to return information about exact matches
            
        Returns:
            If return_exact_match is True:
                tuple: (similar_values, has_exact_match)
                    - similar_values: List of existing values that are similar to any new value
                    - has_exact_match: Boolean indicating if any similarity is exactly 1.0
            Otherwise:
                list: List of existing values that are similar to any new value
        """
        # Skip if either list is empty
        if not new_values or not existing_values:
            if return_exact_match:
                return [], False
            else:
                return []
        
        # Create a combined list for vectorization
        all_values = new_values + existing_values
        
        # Use TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        try:
            # Convert to strings if not already
            all_values_str = [str(v).lower() for v in all_values]
            
            # Create TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_values_str)
            
            # Compute similarity between new and existing values
            new_count = len(new_values)
            new_matrix = tfidf_matrix[:new_count]
            existing_matrix = tfidf_matrix[new_count:]
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(new_matrix, existing_matrix)
            
            # Find existing values that are similar to any new value
            max_similarities = similarity_matrix.max(axis=0)
            similar_indices = np.where(max_similarities >= threshold)[0]
            similar_values = [existing_values[i] for i in similar_indices]
            
            # Check for exact matches (similarity = 1.0)
            has_exact_match = bool(np.any(np.isclose(max_similarities, 1.0)))
            
            if return_exact_match:
                return similar_values, has_exact_match
            else:
                return similar_values
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            if return_exact_match:
                return [], False
            else:
                return []
                
    def _compute_similarity_with_exact_match(self, new_values, existing_values, threshold):
        """
        Compute similarity between new values and existing values, separating exact matches.
        
        Args:
            new_values (list): List of new values
            existing_values (list): List of existing values
            threshold (float): Similarity threshold
            
        Returns:
            tuple: (similar_values, exact_values)
                - similar_values: List of existing values that are similar but not exact matches
                - exact_values: List of existing values that are exact matches (similarity = 1.0)
        """
        # Skip if either list is empty
        if not new_values or not existing_values:
            return [], []
        
        # Create a combined list for vectorization
        all_values = new_values + existing_values
        
        # Use TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        try:
            # Convert to strings if not already
            all_values_str = [str(v).lower() for v in all_values]
            
            # Create TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_values_str)
            
            # Compute similarity between new and existing values
            new_count = len(new_values)
            new_matrix = tfidf_matrix[:new_count]
            existing_matrix = tfidf_matrix[new_count:]
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(new_matrix, existing_matrix)
            
            # Get maximum similarity for each existing value
            max_similarities = similarity_matrix.max(axis=0)
            
            # Find exact matches (similarity = 1.0)
            exact_indices = np.where(np.isclose(max_similarities, 1.0))[0]
            exact_values = [existing_values[i] for i in exact_indices]
            
            # Find similar but not exact matches
            similar_indices = np.where((max_similarities >= threshold) & ~np.isclose(max_similarities, 1.0))[0]
            similar_values = [existing_values[i] for i in similar_indices]
            
            return similar_values, exact_values
        except Exception as e:
            print(f"Error computing similarity with exact match: {str(e)}")
            return [], []

    def upsert_persona(self, persona_json, persona_id):
        """
        Upserts a Persona node using the given persona_id.
        If the Persona exists, update its attribute relationships (by deleting them first).
        Otherwise, create a new Persona node.
        Then, upsert attribute nodes and create relationships from the Persona to each Attribute.
        
        Special case: 'utterances' is a special category that's always allowed and handled separately.
        """
        self.validate_json(persona_json)
        
        with self.driver.session() as session:
            result = session.run("MATCH (p:Persona {id: $id}) RETURN p", id=persona_id).single()
            if result:
                # Update: remove all existing attribute relationships.
                session.run("MATCH (p:Persona {id: $id})-[r]->() DELETE r", id=persona_id)
            else:
                # Insert: create the Persona node.
                session.run("CREATE (p:Persona {id: $id})", id=persona_id)
            
            # Handle special utterances category if it exists
            if 'utterances' in persona_json:
                utterances = persona_json['utterances']
                if utterances:  # Only process if not empty
                    # Store utterances directly as a property on the Persona node
                    session.run(
                        """
                        MATCH (p:Persona {id: $id})
                        SET p.utterances = $utterances
                        """,
                        id=persona_id, utterances=utterances
                    )
            
            # Find field-based categories by checking if value is dict
            field_based_categories = []
            for category in persona_json:
                if isinstance(persona_json[category], dict):
                    field_based_categories.append(category)
                    
            # Process field-based categories (can be basics, demographics, profile, etc.)
            for field_category in field_based_categories:
                if field_category in self.relationship_map:
                    field_data = persona_json.get(field_category, {})
                    rel_type = self.relationship_map[field_category]
                    
                    # Process each field in the category
                    for field, value in field_data.items():
                        if value is None:
                            continue
                        elif isinstance(value, list):
                            # Handle list values for a field
                            for list_item in value:
                                if list_item is not None:
                                    clean_item = str(list_item).strip()
                                    if clean_item:
                                        attr_id = self.compute_attribute_id(field_category, clean_item, key=field)
                                        session.run(
                                        f"""
                                        MERGE (a:Attribute {{id: $attr_id}})
                                        ON CREATE SET a.category = $category, a.key = $field, a.value = $value
                                        ON MATCH SET a.category = $category, a.key = $field, a.value = $value
                                        WITH a
                                        MATCH (p:Persona {{id: $id}})
                                        MERGE (p)-[:{rel_type}]->(a)
                                        """,
                                        attr_id=attr_id, category=field_category, field=field, 
                                        value=clean_item, id=persona_id
                                        )
                            continue
                        
                        # Handle single string values
                        clean_value = str(value).strip()
                        if clean_value:
                            attr_id = self.compute_attribute_id(field_category, clean_value, key=field)
                            session.run(
                                f"""
                                MERGE (a:Attribute {{id: $attr_id}})
                                ON CREATE SET a.category = $category, a.key = $field, a.value = $value
                                ON MATCH SET a.category = $category, a.key = $field, a.value = $value
                                WITH a
                                MATCH (p:Persona {{id: $id}})
                                MERGE (p)-[:{rel_type}]->(a)
                                """,
                                attr_id=attr_id, category=field_category, field=field, 
                                value=clean_value, id=persona_id
                            )
            
            # Process list-based categories
            for category in persona_json:
                # Skip field-based categories and utterances (already processed)
                if category in field_based_categories or category == 'utterances':
                    continue
                
                values = persona_json[category]
                if not isinstance(values, list):
                    values = [values]
                
                # Get the relationship type for this category
                if category in self.relationship_map:
                    rel_type = self.relationship_map[category]
                else:
                    # Create a default relationship type if not in the map
                    rel_type = f"HAS_{category.upper()}"
                
                # Process each value in the list
                for value in values:
                    if value is None:
                        continue
                    
                    clean_value = str(value).strip()
                    if clean_value:
                        attr_id = self.compute_attribute_id(category, clean_value)
                        session.run(
                            f"""
                            MERGE (a:Attribute {{id: $attr_id}})
                            ON CREATE SET a.category = $category, a.value = $value
                            ON MATCH SET a.category = $category, a.value = $value
                            WITH a
                            MATCH (p:Persona {{id: $id}})
                            MERGE (p)-[:{rel_type}]->(a)
                            """,
                            attr_id=attr_id, category=category, value=clean_value, id=persona_id
                        )
        
        return persona_id
        
    def get_persona_info(self, user_id):
        """
        Retrieves persona information from the knowledge graph for a specified user.
        
        Args:
            user_id (str): Identifier for the user (e.g., "user1", "user2")
            
        Returns:
            str: A formatted string containing persona information from the knowledge graph
        """
        # Since we don't have direct mapping from user_id to persona_id in this case,
        # we'll fetch the first persona in the database (or a random one)
        # In a real application, you'd want to map user_id to persona_id more explicitly
        
        with self.driver.session() as session:
            # Get the first persona or a specific one by replacing LIMIT 1 with appropriate filtering
            persona_query = """
            MATCH (p:Persona)
            RETURN p.id AS persona_id
            LIMIT 1
            """
            
            persona_result = session.run(persona_query).single()
            
            if not persona_result:
                return "No persona information available."
                
            persona_id = persona_result["persona_id"]
            
            # Get all attributes for this persona
            attributes_query = """
            MATCH (p:Persona {id: $id})-[r]->(a:Attribute)
            RETURN a.category AS category, a.key AS key, a.value AS value, type(r) AS relationship
            ORDER BY a.category, a.key
            """
            
            attributes = session.run(attributes_query, id=persona_id)
            
            # Organize attributes by category
            attribute_info = {}
            for record in attributes:
                category = record["category"]
                key = record["key"]
                value = record["value"]
                
                if category not in attribute_info:
                    attribute_info[category] = []
                    
                if key:
                    attribute_info[category].append(f"{key}: {value}")
                else:
                    attribute_info[category].append(value)
            
            # Format the information as a readable string
            info_text = f"Knowledge Graph Information for {user_id}:\n\n"
            
            for category, values in attribute_info.items():
                info_text += f"{category.capitalize()}:\n"
                for value in values:
                    info_text += f"- {value}\n"
                info_text += "\n"
            
            return info_text