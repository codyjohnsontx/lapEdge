## Purpose

This file is read automatically by **OpenAI Codex** and other agents (Cursor, Copilot, Aider). Rules here apply to all agent sessions in this repository.

For Claude Code: this repo also has a [`CLAUDE.md`](./CLAUDE.md) which Claude Code reads automatically and contains the same rules.

Agents must follow these instructions to produce safe, predictable, maintainable, minimal changes.

## Instruction Priority

1. Direct user instruction
2. This AGENTS.md
3. Existing repository patterns

## Engineering Standard

Agents must favor readability, correctness, low coupling, explicit boundaries, behavior-preserving refactors, safe data evolution, and minimal surface area over cleverness or speed of implementation.

## Operating Principles

Agents must behave like a careful engineer.

Always:

- read nearby code before editing
- understand the local pattern before changing it
- identify where similar functionality already exists
- make small focused changes
- prefer minimal diffs
- preserve behavior unless the user asked for behavior changes
- stop and report uncertainty, security risk, or missing context instead of guessing
- validate work with the appropriate commands before marking it complete

Never:

- refactor unrelated code
- introduce large rewrites without instruction
- change architecture without instruction
- create parallel implementations when an existing pattern already solves the problem
- optimize prematurely
- hide uncertainty behind confident-sounding code changes

## Diff Size Guardrail

Agents should keep changes small.

Preferred limits:

- fewer than 200 lines changed
- fewer than 5 files modified

If larger work is required:

- explain why
- define the scope
- propose the plan
- wait for approval

## Project Structure

This repository uses a Next.js App Router layout:

- `app/`
- `app/api/`
- `components/`
- `lib/`
- `tests/`
- `public/`
- `content/`
- `sanity/`

Guidelines:

- UI belongs in `components`
- business logic belongs in `lib`
- API handlers belong in `app/api`
- tests belong in `tests`
- static assets belong in `public`
- editable site content belongs in `content`
- Sanity CMS config and schema belong in `sanity`

Do not create new top-level folders unless required.

## Commands

Use these commands to validate work:

```bash
npm run dev
npm run build
npm run lint
npm run test
npx tsc --noEmit
```
