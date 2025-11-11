# Git Workflow and Commit Guidelines

This document describes the git workflow and commit conventions for the OmniDocBench project.

## Conventional Commits

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Scope (Optional)

The scope should indicate the area of the codebase affected:
- `metrics` - Evaluation metrics
- `dataset` - Dataset loaders
- `task` - Evaluation tasks
- `utils` - Utility functions
- `config` - Configuration files
- `cdm` - CDM metric module
- `tools` - Model inference tools

### Examples

```
feat(metrics): add BLEU-4 variant metric

Implement BLEU-4 metric variant for improved text evaluation.
This adds support for 4-gram precision calculation.

Closes #123
```

```
fix(dataset): resolve quick_match edge case for empty predictions

Fixed IndexError when prediction list is empty in quick_match algorithm.
Added boundary checks and empty list handling.
```

```
docs: update CLAUDE.md with new metric instructions

Added section on implementing custom metrics and registry usage.
```

## Code Review with Claude Code

### What is the code-reviewer?
Claude Code has built-in code review capabilities that can perform comprehensive analysis of your changes, checking for security issues, performance problems, code quality, and best practices.

### Recommended Review Process

Before committing, it's **recommended** to use Claude Code's review capabilities:

1. Make your changes
2. Stage files: `git add <files>`
3. In Claude Code, request a review: "Please review my staged changes"
4. Address any issues found by the reviewer
5. Commit with proper conventional commit format

The pre-commit hook will remind you about this step.

## Pre-commit Hooks

The repository uses pre-commit hooks to ensure code quality:

### Installation

```bash
pip install pre-commit
pre-commit install
```

### Hooks Enabled

- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with a newline
- **check-yaml**: Validate YAML syntax
- **check-json**: Validate JSON syntax
- **check-merge-conflict**: Check for merge conflict markers
- **black**: Python code formatting
- **isort**: Sort Python imports
- **flake8**: Python linting

### Running Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run
```

## Git Hooks

### Installation

The git hooks are located in `.git/hooks/` and should be executable:

```bash
chmod +x .git/hooks/prepare-commit-msg
chmod +x .git/hooks/pre-commit
```

### prepare-commit-msg

Validates that commit messages follow conventional commit format. The validation checks:
- Correct type (feat, fix, docs, etc.)
- Optional scope in lowercase with alphanumeric characters, hyphens, or underscores
- Subject starts with lowercase, is 3-50 characters, and has no trailing period

### pre-commit

1. Runs pre-commit hooks (black, isort, flake8, etc.)
2. Provides a reminder about code review recommendations

## Commit Template

A commit template is available at `.gitmessage`. To use it:

```bash
git config commit.template .gitmessage
```

This will provide helpful reminders when writing commit messages.

## Workflow Example

```bash
# 1. Make changes to code
vim metrics/new_metric.py

# 2. Stage changes
git add metrics/new_metric.py

# 3. Run code-reviewer in Claude Code
# (This step is manual - use Claude Code to review)

# 4. Pre-commit hooks run automatically
# Fix any issues if pre-commit fails

# 5. Commit with conventional format
git commit -m "feat(metrics): add new evaluation metric

Implemented CustomMetric for specialized document evaluation.
The metric supports multi-language comparison and handles
edge cases in formula matching.

Closes #456"
```

## Tips

1. **Keep commits atomic**: Each commit should represent a single logical change
2. **Write clear subjects**: Use imperative mood ("add" not "added" or "adds")
3. **Explain why, not what**: The diff shows what changed; explain why in the body
4. **Reference issues**: Link to relevant issues or tickets in the footer
5. **Run tests**: Ensure evaluation tests pass before committing
6. **Use code-reviewer**: Always review with the code-reviewer subagent before committing

## Troubleshooting

### Pre-commit hook fails

```bash
# Fix formatting issues
black .
isort .

# Check specific issues
flake8 --max-line-length=120 <file>

# Re-run hooks
pre-commit run --all-files
```

### Commit message rejected

Ensure your message follows the format:
```
type(scope): subject

body

footer
```

### Skip hooks (emergency only)

```bash
# Not recommended - only for emergencies
git commit --no-verify
```
