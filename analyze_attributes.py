import os
import csv
from knowledge_graph import KnowledgeGraph
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import re
from typing import Dict, List, Set, Tuple
import datetime

class PersonaKGAnalyzer:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def tokenize(self, text: str) -> Set[str]:
        """Convert text to lowercase and split into words, removing special characters."""
        return set(re.findall(r'\w+', text.lower()))

    def word_overlap_ratio(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two strings."""
        words1 = self.tokenize(text1)
        words2 = self.tokenize(text2)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def string_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def get_attributes_data(self) -> Dict[str, List[str]]:
        """Get all attributes from the knowledge graph."""
        return self.kg.get_existing_attributes()
    
    def get_attribute_persona_map(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Get a mapping of categories to {attribute: set of persona_ids}.
        This shows which personas share each attribute.
        """
        with self.kg.driver.session() as session:
            # Get all personas first
            result = session.run("MATCH (p:Persona) RETURN p.id as persona_id")
            self.all_personas = {record["persona_id"] for record in result}
            
            # Get all attribute relationships
            result = session.run("""
                MATCH (p:Persona)-[r]->(a:Attribute)
                RETURN p.id as persona_id, a.category as category, 
                       a.value as value, a.key as key
            """)
            
            # Structure: category -> {attribute -> set(persona_ids)}
            attribute_map = defaultdict(lambda: defaultdict(set))
            
            for record in result:
                category = record["category"]
                value = record["value"]
                key = record["key"]
                persona_id = record["persona_id"]
                
                # For demographics, include the key in the attribute
                if category == "demographics" and key:
                    attribute = f"{key}: {value}"
                else:
                    attribute = value
                    
                attribute_map[category][attribute].add(persona_id)
            
            return attribute_map

    def find_similar_attributes(self, 
                               attributes: Dict[str, List[str]],
                               word_overlap_threshold: float = 0.5,
                               string_sim_threshold: float = 0.8) -> Dict[str, List[Tuple[str, str, float, float]]]:
        """
        Find similar attributes within each category.
        Returns a dictionary of category -> list of (attr1, attr2, word_overlap, string_sim) tuples.
        """
        similar_pairs = defaultdict(list)
        
        for category, values in attributes.items():
            # Skip demographics as they are usually more structured
            if category == "demographics":
                continue
                
            for i, attr1 in enumerate(values):
                for attr2 in values[i+1:]:
                    word_overlap = self.word_overlap_ratio(attr1, attr2)
                    string_sim = self.string_similarity(attr1, attr2)
                    
                    # If either similarity measure is above threshold, consider them similar
                    if word_overlap >= word_overlap_threshold or string_sim >= string_sim_threshold:
                        similar_pairs[category].append((attr1, attr2, word_overlap, string_sim))
        
        return similar_pairs

    def analyze_attribute_similarity(self, save_to_csv=True):
        """Analyze similarity between attributes."""
        print("\nAnalyzing attribute similarity...")
        
        # Get all attributes
        attributes = self.get_attributes_data()
        
        # Find similar attributes
        similar_pairs = self.find_similar_attributes(attributes)
        
        # Print similarity statistics
        print("\nSimilarity Statistics")
        print("====================")
        
        similarity_stats = []
        total_attributes = 0
        total_similar = 0
        
        for category, values in attributes.items():
            if category == "demographics":
                continue
                
            num_attributes = len(values)
            total_attributes += num_attributes
            
            # Get unique attributes involved in similar pairs
            similar_attrs = set()
            if category in similar_pairs:
                for attr1, attr2, _, _ in similar_pairs[category]:
                    similar_attrs.add(attr1)
                    similar_attrs.add(attr2)
            
            num_similar = len(similar_attrs)
            total_similar += num_similar
            
            similarity_percentage = (num_similar / num_attributes * 100) if num_attributes > 0 else 0
            
            print(f"\n{category}:")
            print(f"  Total attributes: {num_attributes}")
            print(f"  Attributes with similarities: {num_similar}")
            print(f"  Percentage similar: {similarity_percentage:.1f}%")
            if category in similar_pairs:
                print(f"  Number of similar pairs: {len(similar_pairs[category])}")
            
            similarity_stats.append({
                'category': category,
                'total_attributes': num_attributes,
                'similar_attributes': num_similar,
                'similarity_percentage': round(similarity_percentage, 1),
                'similar_pairs': len(similar_pairs.get(category, []))
            })
        
        overall_percentage = (total_similar / total_attributes * 100) if total_attributes > 0 else 0
        print(f"\nOverall Statistics:")
        print(f"  Total attributes (excluding demographics): {total_attributes}")
        print(f"  Total attributes with similarities: {total_similar}")
        print(f"  Overall percentage similar: {overall_percentage:.1f}%")
        
        similarity_stats.append({
            'category': 'OVERALL',
            'total_attributes': total_attributes,
            'similar_attributes': total_similar,
            'similarity_percentage': round(overall_percentage, 1),
            'similar_pairs': sum(len(pairs) for pairs in similar_pairs.values())
        })
        
        # Print detailed results
        print("\nDetailed Similar Pairs Analysis")
        print("==============================")
        
        similar_pairs_list = []
        
        for category, pairs in similar_pairs.items():
            if pairs:
                print(f"\n{category}:")
                print("-" * len(category))
                
                # Sort by combined similarity score (word overlap + string similarity)
                pairs.sort(key=lambda x: x[2] + x[3], reverse=True)
                
                for attr1, attr2, word_overlap, string_sim in pairs:
                    print(f"\nPair:")
                    print(f"  1: {attr1}")
                    print(f"  2: {attr2}")
                    print(f"  Word Overlap: {word_overlap:.2f}")
                    print(f"  String Similarity: {string_sim:.2f}")
                    
                    similar_pairs_list.append({
                        'category': category,
                        'attribute1': attr1,
                        'attribute2': attr2,
                        'word_overlap': round(word_overlap, 2),
                        'string_similarity': round(string_sim, 2),
                        'combined_score': round(word_overlap + string_sim, 2)
                    })

        # Save to CSV if requested
        if save_to_csv:
            # Save similarity statistics
            stats_file = os.path.join(self.results_dir, f"similarity_stats_{self.timestamp}.csv")
            with open(stats_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['category', 'total_attributes', 'similar_attributes', 
                                                    'similarity_percentage', 'similar_pairs'])
                writer.writeheader()
                writer.writerows(similarity_stats)
            
            # Save similar pairs details
            pairs_file = os.path.join(self.results_dir, f"similar_pairs_{self.timestamp}.csv")
            if similar_pairs_list:
                with open(pairs_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['category', 'attribute1', 'attribute2', 
                                                        'word_overlap', 'string_similarity', 'combined_score'])
                    writer.writeheader()
                    writer.writerows(similar_pairs_list)
            
            print(f"\nSimilarity analysis saved to:")
            print(f" - {stats_file}")
            print(f" - {pairs_file}")
        
        return similarity_stats, similar_pairs_list

    def analyze_sharing_patterns(self, save_to_csv=True):
        """Analyze and print sharing patterns for attributes."""
        print("\nAnalyzing attribute sharing patterns...")
        
        attribute_map = self.get_attribute_persona_map()
        total_personas = len(self.all_personas)
        
        print("\nAttribute Sharing Analysis")
        print("========================")
        print(f"\nTotal number of personas in database: {total_personas}")
        
        sharing_stats = []
        sharing_distribution = []
        
        for category in sorted(attribute_map.keys()):
            print(f"\n{category}:")
            print("-" * len(category))
            
            # Get all attributes and their sharing counts
            sharing_counts = {
                attr: len(personas)
                for attr, personas in attribute_map[category].items()
            }
            
            if not sharing_counts:
                print("  No attributes found")
                sharing_stats.append({
                    'category': category,
                    'total_attributes': 0,
                    'unique_attributes': 0,
                    'unique_percentage': 0,
                    'shared_attributes': 0,
                    'shared_percentage': 0,
                    'personas_without_attributes': total_personas,
                    'personas_without_percentage': 100.0,
                    'most_shared_count': 0,
                    'avg_sharing': 0
                })
                continue
            
            # Calculate statistics
            total_attributes = len(sharing_counts)
            unique_attributes = sum(1 for count in sharing_counts.values() if count == 1)
            shared_attributes = sum(1 for count in sharing_counts.values() if count > 1)
            
            # Find personas with no attributes in this category
            personas_with_attributes = set().union(*attribute_map[category].values())
            personas_without_attributes = self.all_personas - personas_with_attributes
            
            # Calculate sharing statistics
            max_shared = max(sharing_counts.values()) if sharing_counts else 0
            avg_shared = sum(sharing_counts.values()) / len(sharing_counts) if sharing_counts else 0
            
            print(f"  Total attributes: {total_attributes}")
            print(f"  Unique attributes (used by only one persona): {unique_attributes} ({unique_attributes/total_attributes*100:.1f}%)")
            print(f"  Shared attributes (used by multiple personas): {shared_attributes} ({shared_attributes/total_attributes*100:.1f}%)")
            print(f"  Personas with no attributes in this category: {len(personas_without_attributes)} ({len(personas_without_attributes)/total_personas*100:.1f}%)")
            print(f"  Most shared attribute appears in: {max_shared} personas")
            print(f"  Average sharing per attribute: {avg_shared:.1f} personas")
            
            sharing_stats.append({
                'category': category,
                'total_attributes': total_attributes,
                'unique_attributes': unique_attributes,
                'unique_percentage': round(unique_attributes/total_attributes*100 if total_attributes else 0, 1),
                'shared_attributes': shared_attributes,
                'shared_percentage': round(shared_attributes/total_attributes*100 if total_attributes else 0, 1),
                'personas_without_attributes': len(personas_without_attributes),
                'personas_without_percentage': round(len(personas_without_attributes)/total_personas*100 if total_personas else 0, 1),
                'most_shared_count': max_shared,
                'avg_sharing': round(avg_shared, 1)
            })
            
            # Distribution of sharing
            print("\n  Sharing distribution:")
            sharing_dist = Counter(sharing_counts.values())
            for num_personas, num_attrs in sorted(sharing_dist.items()):
                print(f"    {num_attrs} attributes are shared by {num_personas} personas")
                sharing_distribution.append({
                    'category': category,
                    'num_personas_sharing': num_personas,
                    'num_attributes': num_attrs
                })
            
            # Most commonly shared attributes
            if shared_attributes > 0:
                print("\n  Most commonly shared attributes:")
                most_shared = sorted([(attr, count) for attr, count in sharing_counts.items() if count > 1], 
                                    key=lambda x: x[1], reverse=True)[:5]  # Top 5
                for attr, count in most_shared:
                    print(f"    '{attr}' - shared by {count} personas")

        # Save to CSV if requested
        if save_to_csv:
            # Save sharing statistics
            stats_file = os.path.join(self.results_dir, f"sharing_stats_{self.timestamp}.csv")
            with open(stats_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['category', 'total_attributes', 'unique_attributes', 
                                                    'unique_percentage', 'shared_attributes', 'shared_percentage',
                                                    'personas_without_attributes', 'personas_without_percentage',
                                                    'most_shared_count', 'avg_sharing'])
                writer.writeheader()
                writer.writerows(sharing_stats)
            
            # Save sharing distribution
            dist_file = os.path.join(self.results_dir, f"sharing_distribution_{self.timestamp}.csv")
            if sharing_distribution:
                with open(dist_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['category', 'num_personas_sharing', 'num_attributes'])
                    writer.writeheader()
                    writer.writerows(sharing_distribution)
            
            print(f"\nSharing analysis saved to:")
            print(f" - {stats_file}")
            print(f" - {dist_file}")
        
        return sharing_stats, sharing_distribution

    def run_all_analyses(self):
        """Run all analyses and save results to CSV files."""
        self.analyze_attribute_similarity()
        self.analyze_sharing_patterns()

def main():
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    kg = KnowledgeGraph(uri="bolt://localhost:7690", user="neo4j", password=neo4j_password)
    
    try:
        analyzer = PersonaKGAnalyzer(kg)
        analyzer.run_all_analyses()
    finally:
        kg.close()

if __name__ == "__main__":
    main()
