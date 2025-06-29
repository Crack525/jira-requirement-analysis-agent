#!/bin/bash

# Files to remove
echo "Removing test and debug files..."
rm -f complete_fix.py comprehensive_test.py debug_json_fix.py direct_fix.py isolated_test.py
rm -f manual_fix.py test_json_fix.py trace_fix.py
rm -f fixed_graph_identify_needs.py fixed_main.py patched_main.py
rm -f JSON_FIX_SOLUTION.md JSON_FIX_SUMMARY.md simple_trace.py

# Cleaning debug output files
echo "Cleaning up debug output files..."
rm -f output/debug_llm_raw_response.txt output/debug_prompt_sent.txt
rm -f output/json_fix_attempt.txt output/json_fix_input.txt
rm -f output/fixed_prompts.py output/robust_json_extract_input.txt
rm -f output/prevprompt.txt output/prevprompt2.txt

echo "Workspace cleaned successfully!"
