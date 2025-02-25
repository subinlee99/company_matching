# Company Matching ML

## Overview
This project aims to match company names between two datasets, `Data1_Sample.csv` and `Data2_Sample.csv`, using machine learning techniques. Specifically, the process involves cleaning company names, vectorizing them with TF-IDF, and calculating cosine similarity to identify the best matches.

## Process
### 1. Data Cleaning
- Remove text within parentheses (e.g., stock symbols like `(NYSE:K)`)
- Eliminate common legal terms (Corp., Inc., Ltd., etc.)
- Strip punctuation and normalize whitespace
- Convert names to lowercase for uniformity

### 2. TF-IDF Vectorization
- Convert cleaned company names into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF)

### 3. Cosine Similarity
- Compute pairwise cosine similarity scores between company names in Data1 and Data2
- Identify the best match for each company in Data1 based on similarity scores
- Apply a threshold (default: 0.75) to filter matches

### 4. Matching
- Companies with a similarity score above the threshold are considered a match
- Matched companies are saved to `Matched_Companies.csv`
- Unmatched companies are saved to `Unmatched_Companies.csv`

## Results
- **Total Matches:** 1219
- **Matching Rate:** 97.36%

## Files
- `Data1_Sample.csv`: Sample data containing company names to be matched
- `Data2_Sample.csv`: Reference data containing company names and Excel Company IDs
- `Matched_Companies.csv`: Output file listing matched companies and their corresponding Excel Company IDs
- `Unmatched_Companies.csv`: Output file listing companies from Data1 with no match in Data2

## Usage
1. Ensure you have the necessary dependencies installed:
```bash
pip install pandas scikit-learn
```
2. Run the script:
```bash
python company_matching_ml.py
```
3. Review the results in the generated CSV files.

## Adjustments
- **Threshold:** To fine-tune matching sensitivity, adjust the `best_match_score` threshold in the script (default: 0.75).
