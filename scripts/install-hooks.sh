#!/bin/bash
#
# Install git hooks from scripts/git-hooks/ to .git/hooks/
#
# Usage: ./scripts/install-hooks.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_SOURCE="$PROJECT_ROOT/scripts/git-hooks"
HOOKS_TARGET="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks..."

# Check if .git directory exists
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "Error: Not a git repository"
    exit 1
fi

# Copy hooks
for hook in "$HOOKS_SOURCE"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        echo "  Installing $hook_name..."
        cp "$hook" "$HOOKS_TARGET/$hook_name"
        chmod +x "$HOOKS_TARGET/$hook_name"
    fi
done

echo "✅ Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
ls -1 "$HOOKS_SOURCE"
