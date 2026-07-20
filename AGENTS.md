# AGENTS.md

## Project Overview
This repository is managed by Blackbox CLI v2 with Hermes Agent orchestration.

## Build & Test
- Detect the project stack automatically (Node.js, Python, Go, etc.)
- Run the appropriate build/test commands before committing changes
- Use `npm run build` / `npm test` for Node.js projects
- Use `pytest` / `python -m pytest` for Python projects

## Code Style
- Follow existing conventions in the codebase
- No mock data — all tests use real API calls and real services
- Commit messages: `type: concise description` (types: fix, feat, refactor, docs, chore)

## MCP Servers
- GitHub MCP is configured globally at ~/.blackbox/mcp.json
- Use GitHub MCP tools for issue triage, PR creation, branch management

## Model
- Primary: z-ai/glm-5.2 (78.1% Terminal Bench 2.1, 424.6 t/s)
- Always pass `-m z-ai/glm-5.2` explicitly

## Workflow
1. Read the issue/requirements
2. Understand the codebase structure
3. Implement the fix/feature
4. Run tests (real, not mocks)
5. Commit with descriptive message
6. Push and create PR referencing the issue
