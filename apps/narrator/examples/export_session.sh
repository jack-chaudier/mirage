#!/usr/bin/env bash
# Export a narrator session log as JSON.
# Usage: ./export_session.sh <session_id> [host]
#
# Example:
#   ./export_session.sh abc-123-def > sample_session.json

set -euo pipefail

SESSION_ID="${1:?Usage: $0 <session_id> [host]}"
HOST="${2:-http://localhost:8000}"

curl -s "${HOST}/api/game/${SESSION_ID}/log" | python3 -m json.tool
