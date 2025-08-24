# Claude Assistant Guidelines for Albumentations.ai Blog

## Project Overview

This is the technical blog for Albumentations.ai, focusing on computer vision, image augmentation, and deep learning. The blog serves as the primary channel for:
- Release announcements for Albumentations and AlbumentationsX
- Technical tutorials and guides
- Performance benchmarks
- Community contributions
- Development updates

## Content Standards

### Mathematical Notation
- **ALWAYS use LaTeX format** for mathematical content in markdown files:
  - Inline math: `$formula$` (NOT backticks)
  - Block math: `$$formula$$` (NOT ``` ```math```)
  - Use proper LaTeX commands: `\frac{}{}`, `\partial`, `\sum`, `\sqrt{}`, `\begin{bmatrix}`
  - All subscripts/superscripts: `$x_i$`, `$w_1$`, `$h^2$`
- Mathematical content must be compatible with Next.js MDX rendering (remark-math + rehype-katex)

### Writing Style
- **Technical accuracy is paramount** - never compromise correctness for simplicity
- **Clear distinction** between proven facts and empirical observations
- Use phrases like "we believe", "we assume", "empirically" for unproven claims
- **No toxic or condescending tone** - avoid clichés like "uncovering truths no one talks about"
- Be casual but professional
- Treat readers as experts - no unnecessary hand-holding

### Code Examples
- **Python code** follows PEP 8 style guide
- **Albumentations examples** should use current API (no deprecated `always_apply` parameter)
- Code blocks must have language specification:
  ```python
  # Python code
  ```
  ```bash
  # Shell commands
  ```
- All code examples should be **runnable** - include necessary imports
- Use descriptive variable names
- Add comments for complex operations

### Blog Post Structure

#### Required Frontmatter
```yaml
---
title: string          # Descriptive, SEO-friendly title
date: YYYY-MM-DD      # Publication date
author: string        # Author ID from authors/
categories: [string]  # Categories: announcements, tutorials, performance, community
tags: [string]        # Specific tags
excerpt: string       # 1-2 sentence summary
---
```

#### Content Organization
1. **Hook in introduction** - why should readers care?
2. **Clear structure** with logical progression
3. **No unnecessary repetition** - each section adds value
4. **Depth indicators** (🏊‍♂️) for complex posts to help readers choose their level
5. **Practical examples** and visual aids where helpful
6. **Strong conclusion** with clear takeaways

### Directory Structure
```
posts/
├── YYYY/
│   └── MM-slug/
│       ├── index.md    # Main content
│       └── images/     # Post-specific images
```

### Image Guidelines
- **Hero images**: 1200x630px (Open Graph optimal)
- **In-post images**: Max 1200px width
- **Always include alt text** for accessibility
- Prefer WebP format, PNG/JPG as fallback
- Compress all images

## Technical Review Criteria

### For Mathematical Content
1. **Verify all equations** are mathematically correct
2. **Check LaTeX syntax** renders properly
3. **Ensure notation consistency** throughout the post
4. **Validate mathematical proofs** when presented
5. **Distinguish clearly** between:
   - Proven theorems (with citations)
   - Empirical observations
   - Conjectures or hypotheses

### For Code Content
1. **Test all code snippets** for syntax errors
2. **Verify Albumentations usage** is current and correct
3. **Check imports** are complete
4. **Ensure examples are practical** and relevant
5. **Validate performance claims** with benchmarks

### For General Content
1. **No factual errors** or misleading statements
2. **Proper attributions** and citations
3. **Clear target audience** (beginners, practitioners, researchers)
4. **SEO considerations** without keyword stuffing
5. **Cross-references** to related posts where relevant

## Specific Rules for PRs and Issues

### When responding to @claude mentions:
1. **Always review the entire context** before responding
2. **For implementation requests**: Create complete, working code
3. **For bug fixes**: Identify root cause and provide comprehensive fix
4. **For questions**: Provide specific, actionable answers

### When creating PRs:
1. **Write clear PR descriptions** explaining what and why
2. **Include tests** for new functionality where applicable
3. **Update documentation** if changing existing behavior
4. **Follow existing code patterns** in the repository

## Albumentations-Specific Guidelines

### Library Usage
- Use the latest Albumentations API
- Show both simple and advanced usage patterns
- Include performance comparisons where relevant
- Demonstrate integration with popular frameworks (PyTorch, TensorFlow)

### Common Patterns
```python
import albumentations as A

# Prefer Compose with list of transforms
transform = A.Compose([
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    )
])

# Apply to image
transformed = transform(image=image)
```

### Performance Claims
- Always benchmark with realistic datasets
- Compare against relevant alternatives
- Provide reproducible benchmark code
- State hardware/software configuration

## Repository-Specific Context

### Key Topics
- Image augmentation techniques
- Computer vision best practices
- Deep learning optimization
- Input normalization strategies
- Performance optimization
- Library announcements and updates

### Target Audience
- Machine learning practitioners
- Computer vision researchers
- Kaggle competitors
- Open source contributors
- Data scientists working with images

### Related Projects
- Albumentations (main library)
- AlbumentationsX (commercial version)
- Integration with torchvision, imgaug, etc.

## Memory and Context

### Remember These Preferences
- User prefers extending existing test suites over creating standalone test files
- Use existing methods in functional modules rather than reimplementing
- Avoid unsupported claims - use soft phrasing for unproven statements
- No condescending tone or clichés

### Common Issues to Check
1. Normalization values should be explicit about when division by 255 occurs
2. Mathematical notation must use LaTeX, not Unicode subscripts
3. Code examples should include all necessary imports
4. Distinguish between training from scratch vs fine-tuning

## Response Priorities

When reviewing or implementing:
1. **Correctness** - Technical accuracy above all
2. **Clarity** - Clear, understandable explanations
3. **Completeness** - Full context and runnable examples
4. **Consistency** - Follow existing patterns and style
5. **Performance** - Efficient solutions where it matters

## Quick Reference

### Do's ✅
- Use LaTeX for all math
- Provide complete, runnable code
- Cite sources for claims
- Test before suggesting
- Be specific and actionable
- Preserve mathematical rigor
- Include practical examples

### Don'ts ❌
- Use backticks for math
- Provide partial code snippets
- Make unsupported claims
- Guess at implementations
- Be vague or hand-wavy
- Oversimplify complex topics
- Use deprecated APIs

---

*This document guides Claude's behavior when interacting with the Albumentations.ai blog repository. It ensures consistent, high-quality contributions that maintain the technical standards and style of the project.*
