# Business Requirement Breakdown Agent

A Python Proof-of-Concept (PoC) agent that takes a business requirement, breaks it down into Needs, Epics, and Stories using Google Vertex AI's Gemini model, and outputs the result as a Markdown file. The agent is structured using a conceptual Agent Development Kit (ADK) pattern, with modular tools and a sequential graph workflow.

## Features
- Converts business requirements into structured Needs, Epics, and Stories
- Outputs a well-formatted Markdown document
- Robust error handling for LLM JSON parsing issues
- Includes acceptance criteria for each story
- Modular and extensible architecture

## Project Structure
- `main.py`: Entry point, runs the agent
- `src/agent.py`: Orchestrates the workflow
- `src/graph.py`: Defines the ADK graph and node logic
- `src/tools.py`: LLM and file tools
- `src/prompts.py`: Prompt and Markdown templates
- `src/json_fix.py`: Utilities for fixing LLM JSON errors
- `output/`: Generated markdown files and debug logs

## Setup & Installation
1. **Python 3.9+ required**
2. **Clone the repository**
3. **Set up a virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```
4. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
5. **Configure Google Vertex AI**
   - Enable Vertex AI API in your GCP project
   - Authenticate:
     ```sh
     gcloud auth application-default login
     gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>
     ```
   - Set environment variables in `.env`:
     ```
     GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
     GOOGLE_CLOUD_LOCATION=us-central1
     ```

## Usage
1. Edit `main.py` to provide your business requirement input.
2. Run the agent:
   ```sh
   python main.py
   ```
3. Find the output Markdown file in the `output/` directory.


## License
This project is for demonstration and research purposes only.
