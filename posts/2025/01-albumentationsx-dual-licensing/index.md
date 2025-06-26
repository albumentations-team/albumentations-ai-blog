---
title: "AlbumentationsX: A Fork with Dual Licensing"
date: 2025-06-25
author: vladimir-iglovikov
categories: 
  - announcements
tags:
  - albumentationsx
  - agpl
  - release
excerpt: "AlbumentationsX is a fork of the Albumentations library with dual licensing (AGPL/Commercial). Learn why the change was made and how it affects open-source and commercial users."
featured: true
---

# AlbumentationsX: A Fork with Dual Licensing

AlbumentationsX is a fork of the Albumentations library with dual licensing (AGPL/Commercial). Learn why the change was made and how it affects open-source and commercial users.

## What Happened

- **Albumentations** (MIT license) continues to exist but is no longer actively maintained
- **AlbumentationsX** is a fork with dual licensing (AGPL/Commercial) and active development
- 100% drop-in replacement - same imports, same API, no code changes required

## The Motivation

Albumentations has grown into one of the most widely adopted image augmentation libraries in the world — used by Google, Meta, NVIDIA, Apple, and others. 

Over the past year, I've been the primary maintainer. While the original team members have moved on to new opportunities, the project's usage and support needs have continued to grow:

- Issues, questions, and feature requests arrive weekly
- Major companies rely on this library in production systems
- I worked on this full-time for the past year
- Donations covering only 2.75% of monthly personal expenses — and just 0.39% of prior full-time salary

I haven't been able to make the donation model work for this project. Part of this is structural - companies have established processes for purchasing licenses but not for donations - but ultimately, I wasn't able to build the kind of sponsorship support needed to work on this full-time.

## What Are Your Options?

### Your Three Paths

1. **Continue using albumentations (MIT)** - No restrictions, no fees, but no new updates or bug fixes
2. **Use AlbumentationsX for free** - But you must license your project under AGPL
3. **Purchase a commercial license** - Use AlbumentationsX without open-source restrictions

> ⚠️ **AGPL Reminder**
> 
> AGPL is a viral license. If your project uses MIT, Apache 2.0, or BSD — even if it's open source — you cannot use AlbumentationsX under AGPL. You need a commercial license.

### Understanding AGPL

AGPL extends GPL to network services. What this means:
- If you use AGPL software in a web service, you must make your source code available to users
- If you modify the code, you must share those modifications
- Your entire project must be AGPL-licensed (viral license)
- This applies even if users only interact through a network API

### Real-World Examples

| Your Situation | What You Need |
|---|---|
| Research paper with AGPL code | AlbumentationsX with AGPL ✓ |
| Research paper with MIT/Apache code | Commercial license required |
| Open-source project with MIT license | Commercial license required |
| Company's proprietary ML pipeline | Commercial license required |
| Personal project, not sharing code | Commercial license required |
| Web service using the library | Commercial license (unless AGPL) |

### Quick Comparison

| | albumentations (original) | AlbumentationsX |
|---|---|---|
| **License** | MIT | Dual: AGPL / Commercial |
| **Active maintenance** | No | Yes |
| **New features** | No | Yes |
| **Bug fixes** | No | Yes |
| **Python support** | 3.9-3.13 | 3.9+ with active updates |
| **Code changes required** | - | None |

## How to Switch

> ✅ **Good News**
> 
> AlbumentationsX is a 100% drop-in replacement. Zero code changes required.

If you decide to use AlbumentationsX:

```bash
pip uninstall albumentations
pip install albumentationsx
```

Your existing code continues to work:

```python
import albumentations as A
# Everything works exactly the same
```

## Links

- [Pricing](https://albumentations.ai/pricing) for commercial licenses
- [Discord](https://discord.gg/albumentations) for questions
- [Newsletter](https://albumentations.ai/newsletter) for updates 