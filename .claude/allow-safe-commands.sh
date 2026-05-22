#!/bin/bash
# Auto-allow read-only and test-running Bash commands without a permission prompt.
#
# Matched patterns:
#   - Read-only git:  status, diff, log, show
#   - File listing:   ls, grep
#   - Test runs and linting:
#       conda run -n auto-sheet-music pytest ...
#       conda run -n auto-sheet-music ruff check ...
#     Optionally prefixed by any number of `VAR=value ` env-var assignments
#     (e.g. ATS_PITCH_BACKEND=basic_pitch, PYTHONPATH=/path).
#
# Deliberately NOT matched (still prompts):
#   - conda run -n auto-sheet-music python ...   (arbitrary code execution)
#   - conda run -n auto-sheet-music ruff format  (mutates files)
#   - Anything with && or | (would let unsafe ops piggyback)

cmd=$(jq -r '.tool_input.command // ""')

# Build the regex in parts for readability.
env_prefix='([A-Z_][A-Z0-9_]*=[^ ]+ )*'   # zero or more VAR=value assignments
conda_test="${env_prefix}conda run -n auto-sheet-music (pytest|ruff check)( |$)"
git_read='git (status|diff|log|show)'
listing='ls( |$)|grep( |$)'

if echo "$cmd" | grep -qE "^(${git_read}|${listing}|${conda_test})"; then
  printf '{"hookSpecificOutput":{"hookEventName":"PermissionRequest","permissionDecision":"allow"}}'
fi
