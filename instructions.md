# Benchmarking Document Reading Capabilities
The objective of this repo is to benchmark document reading capabilities of open source models.

# Dataset
Will try to tune more towards financial documents.
1. Primary Benchmark Source: https://huggingface.co/datasets/llamaindex/ParseBench
2. Additionally will add self-supplied documents that are manually verified and corrected.

# Testing Scope
The parsing will be done across multiple document types:
1. Simple text document reading
2. Text aligned randomly on page (common for board of directors page in annual reports of companies for example.)
3. Text with tables & charts
4. Text with infographics
5. Text with images

The models used will be
1. Text-only extraction
2. Text + OCR extraction

# Setup
Multiple experiments will be set up and codebase needs to have simple configs for running experiments. The experiments can be: 
1. Introducing new model for above scope
2. Testing retrieval
3. Testing rerankers