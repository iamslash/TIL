# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **TIL (Today I Learned)** personal knowledge base - a collection of technical documentation and learning notes organized by topic. It contains 390+ topic directories covering programming languages, frameworks, systems, data science, and more.

## Repository Structure

```
/{topic}/
  ├── README.md              # Main documentation for the topic
  ├── {topic}-*.md           # Additional detailed notes
  ├── img/                   # Topic-specific images and diagrams
  └── src/                   # Optional source code examples
```

Topics are organized as flat directories at the root level (e.g., `kubernetes/`, `statistics/`, `python/`).

## Content Conventions

### Markdown Format
- Table of contents at the top using markdown links
- **Materials** section with curated learning resources (courses, videos, blogs)
- Hierarchical sections with clear headings
- Bilingual content (Korean and English mixed)

### LaTeX and Math
- Inline LaTeX formulas embedded as images
- Use fenced code blocks with `latex` language for formula source
- Images generated via external tools (CodeCogs, latex2png)

### Images
- Store in topic-specific `img/` subdirectory
- Reference with relative paths: `![](img/diagram.png)` or `![](filename.png)`
- Use VSCode Paste Image plugin (Cmd+Opt+V) for quick screenshots

## Useful Resources (from README)
- [detexify](http://detexify.kirelabs.org/classify.html) - Draw symbol to get LaTeX code
- [CodeCogs LaTeX editor](https://www.codecogs.com/latex/eqneditor.php) - Generate LaTeX images
- [latex2png](http://latex2png.com/) - Convert LaTeX to PNG
- [Desmos](https://www.desmos.com/) - Interactive graphing

## When Editing Content

- Preserve existing structure and bilingual style
- Add to existing topics rather than creating duplicates
- Maintain the Materials section format with descriptive annotations
- Keep table of contents synchronized with headings
