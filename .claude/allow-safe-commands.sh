#!/bin/bash
# Auto-allow read-only and test-running Bash commands without a permission prompt.
cmd=$(jq -r '.tool_input.command // ""')
if echo "$cmd" | grep -qE '^(git (status|diff|log|show)|ls( |$)|grep( |$)|conda run -n auto-sheet-music (pytest|ruff check))'; then
  printf '{"hookSpecificOutput":{"hookEventName":"PermissionRequest","permissionDecision":"allow"}}'
fi
