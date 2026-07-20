# Changelog

All notable changes to EchoBrief are documented here.
This file is generated from [conventional commits](https://www.conventionalcommits.org) by [git-cliff](https://git-cliff.org).

## [0.2.0] - 2026-07-20

### Documentation

- Add demo gif
- Add docker pull quickstart as recommended try-it path
- Add contributor terms and license/relicensing note

### Features

- Add optional ElevenLabs voice output

### Other

- Remove em dashes from streamlit ui strings
## [0.1.1] - 2026-07-19

### CI/CD

- Build multi-arch image for amd64 and arm64

### Chores

- Bump version to 0.1.1 and update changelog

### Documentation

- Add git-cliff changelog generation
## [0.1.0] - 2026-07-19

### CI/CD

- Automate releases with ghcr image push and github release on tag
- Allow pre-release tags in guard and keep latest stable-only
- Set up buildx so gha cache export works

### Chores

- Scaffold project with uv, tooling config, and CI

### Documentation

- Add model evaluation and final readme polish

### Features

- Add env-driven settings and brief/transcript schemas
- Add faster-whisper transcription service with bundled clip integration test
- Add prompt templates and insight engine with retry and raw-text fallback
- Add segment-boundary chunking with map-reduce for long transcripts
- Add ollama health check, markdown export, and end-to-end CLI
- Add streamlit ui with model selector, progress stages, and markdown download
- Add docker path with compose stack and ci image build
- Add streamlit ui with model selector, progress stages, and markdown download
- Add benchmark harness and fill results with measured numbers
- Generalize prompts and positioning to any recorded call
