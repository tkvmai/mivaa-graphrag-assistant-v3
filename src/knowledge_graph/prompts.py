"""Centralized repository for all LLM prompts used in the knowledge graph system."""

# Phase 1: Main extraction prompts
MAIN_SYSTEM_PROMPT = """
Role: You are an AI expert in Entity and Relationship Extraction for Knowledge Graph generation.

Responsibilities:
- Extract meaningful entities from text.
- Identify relationships (triplets) between entities.
- Ensure predicates (relationship names) are extremely concise.

Critical Guidelines:
- Predicates must be maximum 6 words.
- Prefer 2-3 words for clarity and graph readability.
"""

# --- REFINED MAIN_USER_PROMPT (Focused on Core O&G Data) ---
MAIN_USER_PROMPT = """
Your critical task: Read the text below (delimited by triple backticks) and identify ALL Subject-Predicate-Object (S-P-O) relationships relevant to the **Oil & Gas subsurface data management domain**. For EACH relationship, you MUST identify the TYPE for BOTH the subject and the object. Produce a single JSON array containing objects, where EACH object represents one S-P-O triple and MUST include ALL FIVE of the following keys: "subject", "subject_type", "predicate", "object", and "object_type".

Domain Context:
The text relates specifically to the **Upstream Oil and Gas industry**, focusing on **subsurface exploration and production (E&P) data management**. This includes concepts like:
- **Geoscience Data:** Seismic surveys (2D/3D/4D), well logs (LAS, DLIS), geological models, interpretations, core data, fluid samples.
- **Well Data:** Well headers, trajectories, drilling reports, completion details, production data, well tests.
- **Reservoir Data:** Reservoir models, simulation results, fluid properties, pressure data (PVT).
- **Facilities & Equipment:** Platforms, rigs, pipelines, sensors, specific software names (e.g., Petrel, Techlog, ProSource).
- **Operational Data:** Field operations, drilling campaigns, production allocation, HSE reports.
- **Entities:** Wells, Fields, Basins, Blocks, Reservoirs, Formations, Companies (Operators, Service Companies), Persons (Geoscientists, Engineers), Locations (Countries, Regions, Coordinates), Dates, Measurements (with units), Projects, Software, Concepts (e.g., Data Quality, Migration, Governance), Processes, Standards.

Follow these rules METICULOUSLY:

- **MANDATORY FIELDS:** Every JSON object in the output array MUST contain these exact five keys: `subject`, `subject_type`, `predicate`, `object`, `object_type`. NO EXCEPTIONS. If you cannot determine a specific type, use a reasonable default like "Concept", "Value", or "Identifier", but the key MUST be present.
- **Entity Consistency:** Use consistent, lowercase names for entities (e.g., "daman formation" not "Daman Fm.", "prosource" not "ProSource"). Apply common abbreviations where standard (e.g., "pvt" for Pressure-Volume-Temperature).
- **Entity Types:** Identify the type for each subject and object using **Title Case**. Be specific to the O&G domain. Examples:
    - **Data Types:** `SeismicSurvey`, `WellLog`, `GeologicalModel`, `Interpretation`, `CoreData`, `FluidSample`, `WellTest`, `ProductionData`, `DrillingReport`
    - **Physical Entities:** `Well`, `Field`, `Basin`, `Block`, `Reservoir`, `Formation`, `Platform`, `Rig`, `Pipeline`
    - **Organizations/People:** `Company`, `Operator`, `ServiceCompany`, `Person`, `Team`
    - **Abstract/Other:** `Location`, `Date`, `Measurement`, `Unit`, `Project`, `Software`, `Process`, `Concept`, `Standard` (e.g., OSDU, WITSML), `Identifier`, `Value`
    - Be specific (e.g., `"2.5 mstb/d"` object_type: `Measurement`, `"osdu"` object_type: `Standard`).
- **Atomic Terms:** Identify distinct key terms. Break down complex descriptions if possible (e.g., "high-pressure high-temperature well" might yield `(well_name, is_a, high_pressure_well)` and `(well_name, is_a, high_temperature_well)`).
- **Handle Lists:** If the text mentions a list of items related to a subject (e.g., 'clients including A, B, and C', 'platforms X, Y, Z'), create **separate triples** for each item. Example: `(project, has_client, company_a)`, `(project, has_client, company_b)`, etc.
- **Quantitative Achievements:** Extract specific metrics and link them to the relevant entity (e.g., `(data workspace, achieved_revenue, $3m)`, `(petronas project, reduced_data_stores_by, >80%)`). Use predicates like `has_value`, `achieved_metric`, `improved_by`, `reduced_by`.
- **CRITICAL PREDICATE LENGTH:** Predicates MUST be 4-6 words MAXIMUM, ideally 2-3 words. Be concise and use verbs. Examples: `drilled_by`, `located_in`, `has_target`, `uses_software`, `migrated_to`, `achieved_revenue`. Use lowercase with underscores (`snake_case`).
- **Completeness:** Extract ALL identifiable relationships relevant to the **subsurface E&P data management domain**.
- **Standardization:** Use consistent terminology (e.g., use "well log" consistently).
- **Lowercase Values:** ALL text values for `subject`, `predicate`, and `object` MUST be lowercase.
- **No Special Characters:** Avoid symbols like %, @, “, ”, °, etc., in values. Use plain text equivalents (e.g., "degrees c", "percent").

Important Considerations:
- Precision in naming (wells, fields, formations) is key.
- Maximize graph connectedness via consistent naming and relationship extraction.
- Consider the full context of the chunk.
- **ALL FIVE KEYS (`subject`, `subject_type`, `predicate`, `object`, `object_type`) ARE MANDATORY FOR EVERY TRIPLE.**

Output Requirements:
- Output ONLY the JSON array. No introductory text, commentary, or explanations.
- Ensure the entire output is a single, valid JSON array.
- Each object within the array MUST have the five required keys.

Example of the required output structure (Notice all five keys and domain relevance):

[
  {
    "subject": "well_a_12",
    "subject_type": "Well",
    "predicate": "penetrates_formation",
    "object": "nahr umr formation",
    "object_type": "Formation"
  },
  {
    "subject": "xyz_field",
    "subject_type": "Field",
    "predicate": "operated_by",
    "object": "national oil company",
    "object_type": "Operator"
  },
  {
    "subject": "seismic_survey_2022",
    "subject_type": "SeismicSurvey",
    "predicate": "covers_area",
    "object": "block_7",
    "object_type": "Block"
  },
  {
    "subject": "prosource",
    "subject_type": "Software",
    "predicate": "integrates_with",
    "object": "delfi data ecosystem",
    "object_type": "System"
  },
  {
    "subject": "data_migration_project",
    "subject_type": "Project",
    "predicate": "achieved_metric",
    "object": ">80% reduction", # Example metric
    "object_type": "Value"
  }
]

Crucial Reminder: Every single object in the JSON array must strictly adhere to having the `subject`, `subject_type`, `predicate`, `object`, and `object_type` keys. Ensure predicate is `snake_case`.

Text to analyze (between triple backticks):
"""

# Phase 2: Entity standardization prompts
ENTITY_RESOLUTION_SYSTEM_PROMPT = """
You are an expert in entity resolution and knowledge representation.
Your task is to standardize entity names from a knowledge graph to ensure consistency.
"""

def get_entity_resolution_user_prompt(entity_list):
    return f"""
Below is a list of entity names extracted from a knowledge graph. 
Some may refer to the same real-world entities but with different wording.

Please identify groups of entities that refer to the same concept, and provide a standardized name for each group.
Return your answer as a JSON object where the keys are the standardized names and the values are arrays of all variant names that should map to that standard name.
Only include entities that have multiple variants or need standardization.

Entity list:
{entity_list}

Format your response as valid JSON like this:
{{
  "standardized name 1": ["variant 1", "variant 2"],
  "standardized name 2": ["variant 3", "variant 4", "variant 5"]
}}
"""

# Phase 3: Community relationship inference prompts
RELATIONSHIP_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference. 
Your task is to infer plausible relationships between disconnected entities in a knowledge graph.
"""

def get_relationship_inference_user_prompt(entities1, entities2, triples_text):
    return f"""
I have a knowledge graph with two disconnected communities of entities. 

Community 1 entities: {entities1}
Community 2 entities: {entities2}

Here are some existing relationships involving these entities:
{triples_text}

Please infer 2-3 plausible relationships between entities from Community 1 and entities from Community 2.
Return your answer as a JSON array of triples in the following format:

[
  {{
    "subject": "entity from community 1",
    "predicate": "inferred relationship",
    "object": "entity from community 2"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 6 words maximum. Preferably 2-3 words. Never more than 3.
For predicates, use short phrases that clearly describe the relationship.
IMPORTANT: Make sure the subject and object are different entities - avoid self-references.
"""

# Phase 4: Within-community relationship inference prompts
WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = """
You are an expert in knowledge representation and inference. 
Your task is to infer plausible relationships between semantically related entities that are not yet connected in a knowledge graph.
"""

def get_within_community_inference_user_prompt(pairs_text, triples_text):
    return f"""
I have a knowledge graph with several entities that appear to be semantically related but are not directly connected.

Here are some pairs of entities that might be related:
{pairs_text}

Here are some existing relationships involving these entities:
{triples_text}

Please infer plausible relationships between these disconnected pairs.
Return your answer as a JSON array of triples in the following format:

[
  {{
    "subject": "entity1",
    "predicate": "inferred relationship",
    "object": "entity2"
  }},
  ...
]

Only include highly plausible relationships with clear predicates.
IMPORTANT: The inferred relationships (predicates) MUST be no more than 6 words maximum. Preferably 2-3 words. Never more than 3.
IMPORTANT: Make sure that the subject and object are different entities - avoid self-references.
"""


# --- NEW: Text-to-Cypher Prompt (with Fuzzy Matching for Entities & Relationships) ---
TEXT_TO_CYPHER_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher query translator. Your task is to convert natural language questions into precise Cypher queries based on the provided graph schema to retrieve relevant information.

Graph Schema:
{dynamic_schema}

Core Task:
1. Analyze the user's question (provided in the 'User Input' section below) to understand the specific information requested.
2. Identify the key entities (wells, fields, persons, companies, skills, software, etc.) and the desired relationship(s) mentioned or implied.
3. **Use Pre-linked Entities:** Check the 'Pre-linked Entities' section provided in the 'User Input'. This maps mentions from the question to canonical names found in the graph.
4. Construct a Cypher query that retrieves the requested information using the provided **Graph Schema**.
    - Use `MATCH` clauses to specify the graph pattern. Use specific node labels (e.g., `:Well`, `:Formation`, `:Skill`) from the schema if implied by the question (e.g., "Which wells...?").
    - Filter Entities: Use `WHERE` clauses for filtering on entity names:
        - **Priority:** If a 'Canonical Name in Graph' is provided for a mention in the 'Pre-linked Entities' section, **use an exact, case-insensitive match** on that canonical name: `WHERE toLower(node.name) = toLower('canonical_name_provided')`.
        - **Fallback:** If no canonical name is provided for a mention (shows as 'None' in the pre-linked list) or the entity wasn't pre-linked, use fuzzy, case-insensitive matching on the original mention from the question: `WHERE toLower(node.name) CONTAINS toLower('original_mention')`.
    - Filter Relationships:
        - If the question implies a specific action/relationship type (e.g., "who drilled", "where located", "what skills"), match the specific relationship type from the schema if known (e.g., `MATCH (a)-[r:DRILLED_BY]->(b)`). Relationship types in the schema are typically UPPERCASE_SNAKE_CASE.
        - **CRITICAL SYNTAX:** When matching multiple relationship types OR using a variable with a type, use the pipe `|` separator *without* a colon before subsequent types. Correct: `[r:TYPE1|TYPE2]`, `[:TYPE1|TYPE2]`. Incorrect: `[r:TYPE1|:TYPE2]`, `[:TYPE1|:TYPE2]`.
        - If the relationship is less specific or the exact type is unknown (e.g., "connection between", "related to", "tell me about"), match any relationship (`MATCH (a)-[r]-(b)`) and filter on the original predicate text using fuzzy matching: `WHERE toLower(r.original) CONTAINS 'keyword'`. Use keywords extracted from the question. (Note: `r.original` property stores the original text predicate).
    - Use `RETURN` to specify the output, using aliases like `subject`, `predicate`, `object` where appropriate for clarity. Return distinct results if needed (`RETURN DISTINCT ...`).
    - **Prioritize returning specific properties (like `.name`) rather than entire nodes.**
    - **DO NOT use the generic UNION query pattern unless the question is extremely broad like "show all connections for X".** Focus on targeted queries based on the question's intent and the schema.
5. If the question requires information directly from the source text, you can optionally include a match to the `:Chunk` node and return `c.text`.
6. If the question cannot be answered using the provided schema or is too ambiguous, return the exact text "NO_QUERY_GENERATED".
7. When matching variable-length paths like MATCH p=(e)-[r*1..3]->(x), the variable r represents a list of relationships. To access individual relationship properties like type or .original, you must use the path variable (e.g., p) with functions like relationships(p). For example: MATCH p=(e)-[*1..2]->(x) UNWIND relationships(p) AS rel RETURN type(rel), rel.original.

Query Formatting:
- Enclose the final Cypher query in triple backticks ```cypher ... ```.
- Only return the Cypher query or "NO_QUERY_GENERATED". Do not add explanations or comments outside the backticks.

Examples:

User Question: Who drilled well kg-d6-a#5?
Cypher Query:
```cypher
MATCH (operator:Entity)-[:DRILLED_BY]->(well:Entity)
WHERE toLower(well.name) = 'kg-d6-a#5'
RETURN operator.name AS operator
```

User Question: What formations did well B-12#13 penetrate or encounter?
Cypher Query:
```cypher
MATCH (well:Entity)-[r:PENETRATES|ENCOUNTERED]->(formation:Entity)
WHERE toLower(well.name) = 'b-12#13'
RETURN DISTINCT formation.name AS formation
```

User Question: Tell me about the Daman Formation. (General relationship)
Cypher Query:
```cypher
MATCH (e1:Entity)-[r]-(related:Entity)
WHERE toLower(e1.name) = 'daman formation'
OPTIONAL MATCH (e1)-[:FROM_CHUNK]->(c:Chunk)
RETURN e1.name AS subject, type(r) AS type, r.original AS predicate, related.name AS related_entity, c.text AS chunk_text
LIMIT 25
```

User Question: List all wells drilled by Reliance. (Fuzzy entity, specific relationship type)
Cypher Query:
```cypher
MATCH (operator:Entity)-[:DRILLED_BY]->(well:Entity)
WHERE toLower(operator.name) CONTAINS 'reliance' AND (toLower(well.name) CONTAINS 'well' OR toLower(well.name) CONTAINS '#')
RETURN DISTINCT well.name AS well
```

User Question: How is well A22 related to drilling? (Specific entity, fuzzy relationship)
Cypher Query:
```cypher
MATCH (e1:Entity)-[r]-(related:Entity)
WHERE toLower(e1.name) = 'a22' AND toLower(r.original) CONTAINS 'drill'
RETURN e1.name AS subject, type(r) AS type, r.original AS predicate, related.name AS related_entity
LIMIT 25
```

User Question: What is the capital of France?
Output:
NO_QUERY_GENERATED
"""

# User prompt template for Cypher generation, including few-shot examples
GENERATE_USER_TEMPLATE = """You are a Neo4j expert. Given an input question and potentially pre-linked entities, create a syntactically correct Cypher query to run.
Use the provided schema and few-shot examples to guide your query generation.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!

Schema:
{schema}

Few-shot Examples:
{fewshot_examples}

User Input:
{structured_input}

Cypher query:"""

# --- Prompts for Handling Empty Results ---

EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT = """

You are a meticulous Cypher query analyst. Your objective is to diagnose why a given Cypher query returned zero results when executed against a graph database with the specified schema.

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Based *only* on the user's question, the executed Cypher query, and the schema, determine the *single most likely* reason for the empty result. Choose exclusively from these options:

1.  **QUERY_MISMATCH**: The query's structure (nodes, relationships, properties, filters) does not accurately reflect the semantic intent of the user's question given the schema. For example, the query looks for the wrong relationship type, filters on an inappropriate property, or misunderstands the core entities involved in the question.
2.  **DATA_ABSENT**: The query *does* accurately represent the user's question according to the schema, but the specific data requested simply does not exist in the graph. The query structure is appropriate for the question, but the graph lacks the necessary nodes or relationships satisfying the query's conditions.
3.  **AMBIGUOUS**: The user's question is too vague, unclear, or open to multiple interpretations, making it impossible to definitively determine if the query was mismatched or if the data is truly absent based *only* on the provided information.

**Output Format:**
Respond with ONLY ONE of the reason codes: `QUERY_MISMATCH`, `DATA_ABSENT`, or `AMBIGUOUS`. Do not provide any explanation or justification.

"""

EVALUATE_EMPTY_RESULT_USER_PROMPT = """

**Schema:**
{schema}

**User Question:**
{question}

**Executed Cypher Query (Returned 0 results):**
```cypher
{cypher}
```

**Diagnosis Code:**
"""

# --- Prompts for Revising Queries Evaluated as QUERY_MISMATCH ---

REVISE_EMPTY_RESULT_SYSTEM_PROMPT = """

You are an expert Neo4j Cypher query generator specializing in query correction. You are given a user question, a graph schema, and an initial Cypher query that returned zero results. The initial query has been evaluated as a `QUERY_MISMATCH`, meaning it likely failed to accurately capture the user's intent based on the question and schema.

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Rewrite the original Cypher query to create a *revised* query that better aligns with the user's question and the provided schema.

**Revision Guidelines:**
1.  **Analyze Mismatch:** Identify the specific parts of the original query that likely caused the mismatch (e.g., wrong relationship type, incorrect node label, overly strict filtering on `name` property, wrong direction).
2.  **Reflect Intent:** Construct the revised query to target the entities, relationships, and properties implied by the user's question.
3.  **Consider Flexibility:**
    * If appropriate, make filters less strict (e.g., use `toLower(n.name) CONTAINS 'keyword'` instead of `toLower(n.name) = 'exact match'`).
    * If the relationship type was potentially wrong, try matching a more general pattern (`-[r]-`) and potentially filtering on `r.original` if keywords are available in the question.
    * Ensure the `RETURN` clause provides the specific information requested (e.g., return `entity.name` rather than the whole node).
4.  **Schema Adherence:** Ensure the revised query is valid according to the provided schema.
5.  **No Revision Case:** If, after careful analysis, you determine that the original query *was* the most plausible interpretation of the question despite returning no results, or if no meaningful revision seems possible to better capture the intent, then output the exact text `NO_REVISION`.

**Output Format:**
- If a revision is possible, output *only* the revised Cypher query enclosed in triple backticks: ```cypher\n[REVISED QUERY]\n```
- If no revision is appropriate, output *only* the exact text: `NO_REVISION`
- Do not include any explanations or comments outside the query block or the `NO_REVISION` text.

"""

REVISE_EMPTY_RESULT_USER_PROMPT = """

**Schema:**
{schema}

**User Question:**
{question}

**Original Cypher Query (Returned 0 results, evaluated as QUERY_MISMATCH):**
```cypher
{cypher}
```

**Revised Cypher Query or NO_REVISION:**
"""