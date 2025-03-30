# 🧠 Board Gender Diversity (BGD) and Corporate Culture: A Text-Based Analysis

This project investigates the relationship between Board Gender Diversity (BGD) and corporate culture, following methodologies from **Abernethy et al. (2021)** and **Chen (2024)**. It uses **Glassdoor employee reviews** to extract firm-level cultural scores via both **keyword-based (Stata)** and **word embedding-based (Python NLP)** methods. The final output supports further empirical testing of how BGD shapes culture.

---

## 📁 Folder Structure

BGD-Culture-Analysis/ ├── Extracting Cultural Values using Stata.do # Step 1: Stata script for keyword-based culture score calculation ├── Step 2- NLP-based Cultural Value Extraction using Python.py # Step 2: Python script for Word2Vec-based score extraction ├── READ.me # This file (you may rename it to README.md) ├── Sample Code.rtf # Combined notes (optional for GitHub) ├── Data/ # Raw and processed Glassdoor review data


---

## 🧪 Methodology Overview

### 🔹 Step 1: Stata-Based Keyword Frequency Analysis
- Counts culture-related keywords (Teamwork, Respect, Quality, Integrity, Innovation) in **Glassdoor review text**
- Normalizes by total word count to compute scores
- Aggregates scores to form the **Abernethy culture index**
- Output saved as `.csv` for downstream analysis

Script location: `Extracting Cultural Values using Stata.do`

---

### 🔹 Step 2: Python NLP-Based Cultural Value Extraction
- Preprocesses review text (`pros` and `cons`): cleaning, tokenizing, removing stopwords
- Trains separate **Word2Vec** models for pros and cons reviews
- Computes **semantic similarity scores** between seed culture words and vocab
- Extracts top 30 words per culture dimension and calculates **weighted cultural scores**
- Outputs:
  - `top_30_words_*.csv` (for each dimension)
  - `weighted_scores_cultural_dimensions.csv`
  - `aggregate_cultural_scores.csv`

Script location: `Step 2- NLP-based Cultural Value Extraction using Python.py`

---

## 🧩 Data Requirements

Put your raw Glassdoor data inside the `Data/` folder, formatted like:

- `review_pros`: pros text
- `review_cons`: cons text
- `company_name`: firm identifier

⚠️ **Important:** Be sure to preprocess and clean company names consistently for later matching.

---

## 🔍 Optional Step: Fuzzy Matching (included in Python script)

- Standardizes and fuzzy matches firm names between Glassdoor and BoardEx datasets
- Uses `rapidfuzz` for matching
- Outputs merged dataset `matched_glassdoor_boardex.csv` for empirical testing

---

## 📦 Python Dependencies

Install required libraries using:

```bash
pip install pandas numpy gensim scikit-learn rapidfuzz


📊 Outputs
You can find final analysis results (after running the scripts) under:

top_30_words_pros_*.csv and top_30_words_cons_*.csv

weighted_scores_cultural_dimensions_*.csv

aggregate_cultural_scores_*.csv

matched_glassdoor_boardex.csv


👩‍💻 Author
Xinying Fu
MSA Student | Research interests: Corporate Governance,  NLP in Accounting
📚 References
Abernethy, M. A., et al. (2021). Do we understand how management controls organizational culture? Accounting, Organizations and Society.

Chen, Y. (2024). Textual Measures of Culture Using Word Embeddings.
