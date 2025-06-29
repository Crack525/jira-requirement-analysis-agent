# main.py
import os
from dotenv import load_dotenv
import logging
import sys
import traceback
import json

# Assuming your ADK agent code is in the src directory
# Add src to the system path if running directly from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.agent import RequirementBreakdownAgent # Import from the src package

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility functions for error handling
# These imports must come after setting sys.path
from src.json_fix import sanitize_error_message, create_mock_needs

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") # Or your preferred location
OUTPUT_DIR = "./output"

def safely_extract_error(value):
    """Safely extract error information without propagating problematic strings"""
    if isinstance(value, str) and ('\n    "name"' in value or value == '\n    "name"'):
        return "JSON parsing error with name field"
    return str(value)

def continue_with_mock_data(agent):
    """When an error occurs, continue with mock data for testing"""
    logger.info("Continuing workflow with mock data")
    
    # Create mock hierarchy directly to avoid LLM calls
    mock_needs = create_mock_needs()
    
    mock_epic = {
        "name": "Implement Test Qualification System",
        "description": "Create a framework to systematically qualify which tests should run in production",
        "acceptance_criteria": ["Test qualification process defined", "Qualification system implemented", "Cost control measures in place"]
    }
    
    mock_stories = [
        {
            "name": "Define Test Qualification Criteria",
            "description": "Define criteria for determining which tests can safely run in production",
            "acceptance_criteria": ["Documentation created", "Criteria approved by stakeholders"]
        },
        {
            "name": "Implement Test Tagging System",
            "description": "Create a system to tag tests based on whether they can run in production",
            "acceptance_criteria": ["Tagging system implemented", "Tests properly tagged"]
        },
        {
            "name": "Create Cost Control Measures",
            "description": "Implement controls to prevent high cost or performance issues in production tests",
            "acceptance_criteria": ["Cost thresholds defined", "Monitoring in place", "Automatic test termination implemented"]
        }
    ]
    
    # Full hierarchy to be used
    mock_hierarchy = {
        "needs": mock_needs,
        "epics": [mock_epic],
        "stories_by_epic": [
            {
                "epic": mock_epic,
                "stories": mock_stories
            }
        ]
    }
    
    # Create mock state with our prepared data
    mock_state = {
        "business_requirement": SAMPLE_BUSINESS_REQUIREMENT,
        "identify_needs_output": mock_needs,
        "generate_hierarchy_output": mock_hierarchy
    }
    
    # Get the tools needed for execution
    tools = {
        "llm_tool": agent.llm_tool,
        "file_tool": __import__("src.tools", fromlist=["MarkdownFileTool"]).MarkdownFileTool(),
    }
    
    # Import necessary graph component - just need markdown generation now
    from src.graph import generate_markdown_logic

    # Run the markdown step directly
    try:
        logger.info("Running generate_markdown_logic with mock hierarchy")
        markdown_output = generate_markdown_logic(mock_state, {"file_tool": tools["file_tool"]})
        mock_state["generate_markdown_output"] = markdown_output
        
        return {
            "status": "completed_with_mock_data", 
            "final_state": mock_state
        }
    except Exception as e:
        logger.error(f"Error in mock workflow continuation: {e}")
        traceback.print_exc()
        return {
            "status": "failed_mock_continuation",
            "error": str(e),
            "final_state": mock_state
        }

# Sample Business Requirement (from your input)
SAMPLE_BUSINESS_REQUIREMENT = """
Need:
Existing DS Core platform in different production regions needs to be monitored and maintained. Refactoring of existing services are done to ensure performance and security of the platform. Incoming service tickets need to be answered (second level). 

Impact of not doing this:
Negative impact on customer satisfaction of existing DS Core users.
"""

# --- Main Execution ---
if __name__ == "__main__":
    if not PROJECT_ID:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set.")
        logger.error("Please create a .env file at the project root with GOOGLE_CLOUD_PROJECT=your-gcp-project-id")
        exit(1)

    logger.info(f"Starting Agent execution for Project: {PROJECT_ID}, Location: {LOCATION}")

    # Ensure the virtual environment is activated
    # Note: In a real deployment (e.g. on MCP), the environment would be managed.
    # For local VS Code execution, the user must activate it manually before running main.py
    if 'VIRTUAL_ENV' not in os.environ:
        logger.warning("Virtual environment is not activated. Please activate the 'venv' environment before running main.py.")
        # You might want to add a check and exit here in a more robust script
        # exit(1)


    try:
        # Initialize the Agent
        agent = RequirementBreakdownAgent(project_id=PROJECT_ID, location=LOCATION, output_dir=OUTPUT_DIR)

        # Run the Agent with the business requirement
        # Add debug logging to see the input
        print(f"DEBUG: Passing business_requirement: \n{SAMPLE_BUSINESS_REQUIREMENT}\n")
        result = agent.run(business_requirement=SAMPLE_BUSINESS_REQUIREMENT)

        # Report the outcome
        if result["status"] == "completed":
            final_state = result["final_state"]
            # The file path is stored in the state from the generate_markdown_logic node
            output_path = final_state.get('generate_markdown_output', {}).get('file_path')
            if output_path:
                logger.info(f"Agent completed successfully.")
                # Provide the full path for easier access
                full_output_path = os.path.abspath(output_path)
                logger.info(f"Output saved to: {full_output_path}")
            else:
                logger.error("Agent completed, but output file path was not found in the final state.")
        else: # status is "failed"
            logger.error(f"Agent failed during execution.")
            
            # Safely extract error information
            error = safely_extract_error(result.get('error', 'Unknown error'))
            
            # Special handling for the problematic error pattern
            if "JSON parsing error" in error or "name" in error:
                logger.error("JSON parsing error with name field detected")
                
                # Try to continue with mock data
                recovery_result = continue_with_mock_data(agent)
                if recovery_result["status"] == "completed_with_mock_data":
                    logger.info("Successfully recovered from error using mock data!")
                    output_path = recovery_result["final_state"].get('generate_markdown_output', {}).get('file_path')
                    if output_path:
                        full_output_path = os.path.abspath(output_path)
                        logger.info(f"Recovery output saved to: {full_output_path}")
                else:
                    logger.error(f"Recovery attempt failed: {recovery_result.get('error')}")
            else:
                logger.error(f"Failure details: {error}")
                
            logger.error(f"Failed node: {result.get('failed_node', 'N/A')}")

    except RuntimeError as e:
        logger.critical(f"Agent initialization failed: {safely_extract_error(str(e))}")
        exit(1)
    except Exception as e:
        error_msg = safely_extract_error(str(e))
        logger.critical(f"An unhandled exception occurred during agent run: {error_msg}", exc_info=True)
        
        # Write debug info to file
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/critical_error.txt", "w") as f:
                f.write(f"Critical error: {error_msg}\n\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
            
        exit(1)