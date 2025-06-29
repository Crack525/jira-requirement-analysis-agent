import json
import logging
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
from typing import Dict, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Conceptual ADK Tool Base (replace with actual ADK base if available) ---
# This is a simplified representation. In a real ADK SDK, you'd likely inherit
# from a specific AdkTool base class.
class AdkTool:
    """Base class for conceptual ADK tools."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, **kwargs) -> Dict[str, Any]:
        """Abstract method to be implemented by concrete tools."""
        raise NotImplementedError("Subclasses must implement 'run'")

# --- Concrete Tool Implementations ---

class VertexAILLMTool(AdkTool):
    """Tool for interacting with Vertex AI Gemini LLM."""

    def __init__(self, project_id: str, location: str, model_name: str = "gemini-2.0-flash"):
        super().__init__(name="VertexAILLMTool", description="Interacts with Google Vertex AI Gemini models.")
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.model = None # Initialize lazily or within run if needed
        self.initialization_error = None # Store initialization error

        try:
            logger.info(f"Initializing Vertex AI: Project={self.project_id}, Location={self.location}")
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Successfully initialized model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI or model {self.model_name}: {e}")
            # Depending on framework, might raise here or handle later
            # In this PoC, we'll let the graph node handle the tool failure
            self.initialization_error = str(e)


    def run(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 8000) -> Dict[str, Any]:
        """Sends a prompt to the LLM and returns the parsed JSON response."""
        if self.model is None:
             return {"success": False, "error": f"LLM tool not initialized: {self.initialization_error}"}

        # Add a JSON formatting instruction as a prefix to the prompt
        json_instruction = """
You are a JSON response generator. Your task is to:
1. ALWAYS provide responses in strict JSON format
2. ALWAYS enclose the entire response in [] for arrays or {} for objects
3. NEVER return partial JSON fragments
4. NEVER omit the opening or closing brackets/braces
5. ALWAYS use double quotes for keys and string values

Provide a valid, well-structured JSON response following these JSON formatting rules:
"""
        
        # For prompts that specifically request JSON, add the instruction
        if "JSON" in prompt or "json" in prompt:
            enhanced_prompt = json_instruction + "\n\n" + prompt
        else:
            enhanced_prompt = prompt
        
        logger.info(f"Calling LLM with enhanced prompt (first 200 chars): {enhanced_prompt[:200]}...")
        try:
            # Use a lower temperature for JSON responses to improve consistency
            response = self.model.generate_content(
                enhanced_prompt,
                generation_config={
                    "temperature": min(temperature, 0.1),  # Use even lower temperature for better JSON formatting
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",  # Request JSON output format
                }
            )
            response_text = response.text.strip()
            logger.info("[DEBUG] Raw LLM response received. Validating...")
            
            # First, validate the response format before trying to parse it
            def validate_json_format(text):
                """Validate if text appears to be a well-formed JSON structure"""
                # Check if it's completely empty
                if not text or text.isspace():
                    return False, "Empty response"
                    
                # Check for the problematic pattern
                if text == '\n    "name"' or text.strip() == '"name"':
                    logger.error(f"CRITICAL ERROR: Direct match with problematic pattern: {repr(text)}")
                    try:
                        import os
                        os.makedirs("output", exist_ok=True)
                        with open("output/critical_error_pattern.txt", "w") as f:
                            f.write(str(text))
                    except Exception as e:
                        logger.error(f"Error saving critical error pattern: {e}")
                    return False, "Critical error pattern detected"
                
                # Single field name pattern (more generalized)
                import re
                name_pattern = re.compile(r'^\s*["\']?\s*\w+\s*["\']?\s*$')
                if name_pattern.match(text.strip()):
                    logger.error(f"FIELD NAME PATTERN DETECTED: {repr(text)}")
                    return False, "Field name pattern detected"
                
                # Check for basic JSON structure
                text = text.strip()
                if not ((text.startswith('{') and text.endswith('}')) or 
                       (text.startswith('[') and text.endswith(']'))):
                    if '```json' in text or '```' in text:
                        # It might be JSON in a code block, which we can extract later
                        pass
                    else:
                        return False, "Response doesn't appear to have proper JSON structure with brackets/braces"
                
                return True, "Validation passed"
            
            # Validate the JSON format
            is_valid, validation_message = validate_json_format(response_text)
            if not is_valid:
                logger.warning(f"JSON validation failed: {validation_message}")
                # For non-critical failures, we'll still try to parse later
                if "Critical error pattern" in validation_message or "Field name pattern" in validation_message:
                    return {"success": False, "error": validation_message, "raw_response": response_text}

            # Remove code block markers if present
            if response_text.startswith("```json"):
                json_string = response_text[len("```json"):].strip().rstrip("```")
            elif response_text.startswith("```"):
                json_string = response_text[len("```"):].strip().rstrip("```")
            else:
                json_string = response_text
            
            # Fix common JSON formatting issues
            # 1. First, ensure we have actual JSON-like content by looking for { or [
            if not ('{' in json_string or '[' in json_string):
                return {"success": False, "error": "No JSON object or array found in LLM response", "raw_response": response_text}

            # 2. Advanced extraction using regex to find the largest JSON structure
            import re
            # This regex will find JSON-like structures (arrays or objects)
            # Using a more greedy pattern to capture full structures
            matches = re.findall(r'([\[{][\s\S]*?[\]}])', json_string)
            # Also try a more aggressive regex that might better handle nested structures
            if not matches:
                matches = re.findall(r'([\[{][\s\S]*[\]}])', json_string)
            
            if matches:
                # Pick the largest match (most likely the full array/object)
                json_string = max(matches, key=len)
            
            # 3. Clean up: remove any text before the first { or [ and after the last } or ]
            first_bracket = min(json_string.find('{') if '{' in json_string else float('inf'), 
                               json_string.find('[') if '[' in json_string else float('inf'))
            if first_bracket > 0:
                json_string = json_string[first_bracket:]
                
            last_bracket = max(json_string.rfind('}') if '}' in json_string else -1, 
                              json_string.rfind(']') if ']' in json_string else -1)
            if last_bracket > 0:
                json_string = json_string[:last_bracket + 1]
            
            # 4. Handle case where LLM returns incomplete structure by trying to fix common issues
            # Balance brackets - only do this if we have a clear imbalance
            open_curly = json_string.count('{')
            close_curly = json_string.count('}')
            open_square = json_string.count('[')
            close_square = json_string.count(']')
            
            if open_curly > close_curly:
                json_string += '}' * (open_curly - close_curly)
            if open_square > close_square:
                json_string += ']' * (open_square - close_square)
                
            # 5. Final cleanup - ensure whitespace is trimmed
            json_string = json_string.strip()
            
            print("[DEBUG] Enhanced extracted JSON string:", json_string)
            
            try:
                data = json.loads(json_string)
                logger.info("Successfully parsed LLM response as JSON.")
                return {"success": True, "data": data}
            except json.JSONDecodeError as parse_e:
                # If we still fail, try one more fallback with a simpler approach
                logger.error(f"Failed to parse JSON after enhancements: {parse_e}")
                print("[DEBUG] JSONDecodeError after enhancements:", parse_e)
                
                # Convert the error to a proper format - don't return raw Python exception objects
                error_msg = f"JSON Parse Error: {str(parse_e)}"
                
                # Last resort - try to manually build JSON from the response
                # This is for cases where the LLM has returned a mix of text and JSON
                if json_string.strip().startswith('"name"'):
                    # Handle case where JSON is missing opening bracket
                    fixed_json = '{' + json_string.strip()
                    if not fixed_json.endswith('}'):
                        fixed_json += '}'
                    print("[DEBUG] Last resort fix attempt (object):", fixed_json)
                    try:
                        data = json.loads(fixed_json)
                        logger.info("Successfully parsed LLM response using last resort fix (object).")
                        # If this is expected to be an array but we got a single object
                        if '"impact_of_not_doing"' in json_string:  # This is likely a Need object
                            data = [data]  # Wrap in array
                            logger.info("Converted single Need object to array.")
                        return {"success": True, "data": data}
                    except json.JSONDecodeError:
                        pass
                
                # Handle when we see a formatted array but it's missing the array brackets
                if json_string.strip().startswith('"name"') or json_string.strip().startswith('{\n') or json_string.strip().startswith('{\r\n'):
                    # Try to see if this is actually an array item without the outer brackets
                    fixed_json = '[{' + json_string.strip()
                    # Only add closing braces if we determine they're likely needed
                    open_curly = fixed_json.count('{')
                    close_curly = fixed_json.count('}')
                    if open_curly > close_curly:
                        fixed_json += '}' * (open_curly - close_curly)
                    if not fixed_json.endswith(']'):
                        fixed_json += ']'
                    print("[DEBUG] Last resort fix attempt (array):", fixed_json)
                    try:
                        data = json.loads(fixed_json)
                        logger.info("Successfully parsed LLM response using last resort array fix.")
                        return {"success": True, "data": data}
                    except json.JSONDecodeError:
                        pass
                        
                # Additional fallback for LLM responses that have field names at root level
                # Important: Handle the exact problematic pattern early
                if json_string == '\n    "name"' or str(json_string) == '\n    "name"':
                    logger.error(f"[CRITICAL] Detected the exact problematic pattern: {repr(json_string)}")
                    try:
                        # Save the problematic pattern for debugging
                        with open("output/problematic_pattern.txt", "w") as f:
                            f.write(repr(json_string))
                    except Exception as e:
                        logger.error(f"Error saving problematic pattern: {e}")
                    
                    # Return clear error for downstream handlers
                    return {
                        "success": False, 
                        "error": "CRITICAL: Exact problematic pattern detected",
                        "raw_response": json_string,
                        "error_code": "PROBLEMATIC_PATTERN"
                    }
                
                # Handle any response that has field names at root level
                if '\n    "name"' in json_string or '\n  "name"' in json_string:
                    # This looks like our specific error case - the direct fix that works
                    try:
                        # Clean up the response by removing newlines + indentation, but keep fields
                        lines = [line.strip() for line in json_string.split('\n') if line.strip()]
                        cleaned_text = ' '.join(lines)
                        print(f"[DEBUG] Cleaned text: {cleaned_text}")
                        
                        # Fix the opening bracket
                        fixed_json = '[{' + cleaned_text
                        
                        # Make sure the closing is correct
                        if not fixed_json.endswith("}]"):
                            if fixed_json.endswith("}"):
                                fixed_json += "]"
                            else:
                                fixed_json += "}]"
                        
                        print(f"[DEBUG] DIRECT FIX JSON: {fixed_json}")
                        data = json.loads(fixed_json)
                        
                        if isinstance(data, list) and len(data) > 0:
                            logger.info(f"Successfully fixed JSON with direct fix! Found {len(data)} items.")
                            return {"success": True, "data": data}
                    except json.JSONDecodeError as e:
                        print("[DEBUG] Special case fix failed:", e)
                
                # --- FORCE DIRECT FIX FOR ANY ROOT-LEVEL "name" FIELD ---
                if re.search(r'(^|\n)\s*"name"\s*:', json_string):
                    logger.warning("[FORCED FIX] Detected root-level 'name' field. Forcing direct JSON fix.")
                    try:
                        lines = [line.strip() for line in json_string.split('\n') if line.strip()]
                        cleaned_text = ' '.join(lines)
                        fixed_json = '[{' + cleaned_text
                        if not fixed_json.endswith('}]'):
                            if fixed_json.endswith('}'):
                                fixed_json += ']'
                            else:
                                fixed_json += '}]'
                        logger.info(f"[FORCED FIX] Fixed JSON: {fixed_json}")
                        data = json.loads(fixed_json)
                        if isinstance(data, list) and len(data) > 0:
                            logger.info(f"[FORCED FIX] Successfully parsed! Found {len(data)} items.")
                            return {"success": True, "data": data}
                    except Exception as e:
                        logger.error(f"[FORCED FIX] Failed: {e}")
                
                # Try a retry with more explicit instructions
                try:
                    # Only retry if this doesn't match our known problematic patterns
                    if '"name"' not in json_string and not (json_string == '\n    "name"'):
                        logger.info("Attempting retry with more explicit JSON instructions...")
                        retry_instruction = f"""
The previous response could not be parsed as valid JSON. 
Please fix the JSON formatting issues and provide a properly formatted JSON response.

Previous response (with errors):
{json_string}

Please provide a fixed, valid JSON response that follows proper JSON syntax with:
- Opening and closing brackets/braces
- Proper quotes around keys and string values
- Commas between elements
- No trailing commas
- Proper escaping of special characters

RESPOND ONLY WITH THE CORRECTED JSON:
"""
                        retry_response = self.model.generate_content(
                            retry_instruction,
                            generation_config={
                                "temperature": 0.01,  # Ultra low temperature for formatting task
                                "max_output_tokens": max_output_tokens,
                                "response_mime_type": "application/json"
                            }
                        )
                        
                        retry_text = retry_response.text.strip()
                        # Remove code block markers if present
                        if retry_text.startswith("```json"):
                            retry_json = retry_text[len("```json"):].strip().rstrip("```")
                        elif retry_text.startswith("```"):
                            retry_json = retry_text[len("```"):].strip().rstrip("```")
                        else:
                            retry_json = retry_text
                            
                        # Try to parse the retry response
                        retry_parsed = json.loads(retry_json)
                        logger.info("Successfully parsed JSON after retry!")
                        return {"success": True, "data": retry_parsed, "retried": True}
                except Exception as retry_e:
                    logger.error(f"Retry attempt also failed: {retry_e}")

                # Use our specialized fix from json_fix as a last resort
                try:
                    from .json_fix import robust_json_extract
                    logger.info("Attempting robust JSON extraction as last resort...")
                    extracted_data = robust_json_extract(response_text)
                    
                    if extracted_data and isinstance(extracted_data, list) and len(extracted_data) > 0:
                        logger.info(f"Robust JSON extraction successful! Found {len(extracted_data)} items.")
                        return {"success": True, "data": extracted_data, "method": "robust_extraction"}
                except Exception as extract_e:
                    logger.error(f"Robust extraction failed: {extract_e}")

                return {"success": False, "error": "JSON parse error", "error_details": str(parse_e), "raw_response": response_text}

        except Exception as e:
            logger.error(f"Error during LLM generation or processing: {e}")
            return {"success": False, "error": f"LLM generation failed: {e}"}

class MarkdownFileTool(AdkTool):
    """Tool for writing markdown content to a file."""

    def __init__(self):
        super().__init__(name="MarkdownFileTool", description="Writes content to a markdown file.")

    def run(self, content: str, file_path: str) -> Dict[str, Any]:
        """Writes the given content string to a file."""
        logger.info(f"Attempting to write content to {file_path}")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully wrote content to {file_path}")
            return {"success": True, "file_path": file_path}
        except IOError as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return {"success": False, "error": f"File writing failed: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing file {file_path}: {e}")
            return {"success": False, "error": f"Unexpected file error: {e}"}

# --- Helper function to get tools (used in graph definition) ---
def get_tools(project_id: str, location: str) -> Dict[str, AdkTool]:
    """Instantiates and returns the tools for the agent."""
    return {
        "llm_tool": VertexAILLMTool(project_id=project_id, location=location),
        "file_tool": MarkdownFileTool(),
    }