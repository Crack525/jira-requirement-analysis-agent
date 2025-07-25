# src/prompts.py

# Prompt #1: Identify Needs
PROMPT_IDENTIFY_NEEDS = """
Extract key high-level Needs from this business requirement. Each Need should represent a significant business problem, opportunity, or desired outcome.

Business Requirement:
{business_requirement}

IMPORTANT: You MUST respond with VALID JSON that follows the exact structure below.
Always include the opening '[' and closing ']' brackets. 
Always format each object with proper JSON syntax including quotes and commas.
Always use double quotes for property names and string values.

Return your response in this exact JSON array format:
[
  {{
    "name": "Brief name of the need",
    "description": "Short description of what this need entails",
    "impact_of_not_doing": "Brief impact statement"
  }}
]

Example of a correct response:
[
  {{
    "name": "Test Qualification System",
    "description": "A system to qualify which tests can run in production",
    "impact_of_not_doing": "Potential system availability issues"
  }}
]

REMINDER: Your entire response MUST be a valid JSON array, starting with '[' and ending with ']'.
"""

# Prompt #2: Generate Epic from Need
PROMPT_GENERATE_EPIC = """
Create a single Epic to address this Need:

Need:
Name: {need_name}
Description: {need_description}
Impact of not doing: {need_impact}

IMPORTANT: You MUST respond with a VALID JSON object. 
Always include the opening '{{' and closing '}}' braces.
Always use double quotes for property names and string values.

Return your response in this exact JSON object format:
{{
  "name": "Epic title",
  "description": "Epic description",
  "acceptance_criteria": ["Criterion 1", "Criterion 2", "Criterion 3"]
}}

Example of a correct response:
{{
  "name": "Implement Test Qualification Framework",
  "description": "Create a framework to systematically qualify which tests should run in production",
  "acceptance_criteria": ["Qualification criteria defined", "Process documented", "Implementation completed"]
}}

REMINDER: Your entire response MUST be a valid JSON object, starting with '{{' and ending with '}}'.
"""

# Prompt #3: Generate Stories from Epic
PROMPT_GENERATE_STORIES = """
Break down the following Epic into user stories:

Epic:
Title: {epic_name}
Description: {epic_description}

IMPORTANT: You MUST respond with a VALID JSON array containing story objects.
Always include the opening '[' and closing ']' brackets.
Always format each object with proper JSON syntax including quotes and commas.
Always use double quotes for property names and string values.

Return your response in this exact JSON array format:
[
  {{
    "name": "Story title",
    "description": "Story description",
    "acceptance_criteria": ["Criterion 1", "Criterion 2"]
  }},
  {{
    "name": "Another story title",
    "description": "Another story description",
    "acceptance_criteria": ["Criterion 1", "Criterion 2"]
  }}
]

Example of a correct response:
[
  {{
    "name": "Define Test Qualification Criteria",
    "description": "As a developer, I want clear criteria for which tests can run in production",
    "acceptance_criteria": ["Documentation created", "Criteria approved by stakeholders"]
  }},
  {{
    "name": "Implement Test Tagging System",
    "description": "As a tester, I want to tag tests based on their production readiness",
    "acceptance_criteria": ["Tagging system implemented", "All tests properly tagged"]
  }}
]

REMINDER: Your entire response MUST be a valid JSON array, starting with '[' and ending with ']'.
"""

# Markdown template
MARKDOWN_TEMPLATE = """\
# Business Requirement Breakdown

**Original Requirement:**
{original_requirement}

---

## Needs Identified:
{needs_section}

---
*Generated by ADK Agent on {timestamp}*
"""

MARKDOWN_NEED_SECTION_TEMPLATE = """\

### Need: {need_name}
*   **Description:** {need_description}
*   **Impact of Not Doing:** {need_impact}
{epic_section}
"""

MARKDOWN_EPIC_SECTION_TEMPLATE = """\

#### Epic for Need: {need_name}
*   **Title:** {epic_title}
*   **Description:** {epic_description}
{stories_section}
"""

MARKDOWN_STORIES_SECTION_TEMPLATE = """\

##### Stories for Epic: {epic_title}
{stories_list}
"""

MARKDOWN_STORY_TEMPLATE = """\
*   Story: {story_title}
    *   Description: {story_description}
    *   Acceptance Criteria: {story_acceptance_criteria}
"""
