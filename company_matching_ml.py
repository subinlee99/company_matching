import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
data1 = pd.read_csv('Data1_Sample.csv')
data2 = pd.read_csv('Data2_Sample.csv')

print(f'Data1 loaded with {len(data1)} rows')
print(f'Data2 loaded with {len(data2)} rows')

# Updated data cleaning function
def clean_company_name(name):
    if pd.isnull(name):
        return ''
    name = re.sub(r'\(.*?\)', '', str(name))  # Remove text in parentheses
    name = re.sub(r'\b(NYSE:[A-Za-z]+|NASDAQ:[A-Za-z]+)\b', '', name, flags=re.IGNORECASE)  # Remove stock symbols
    name = re.sub(r'\b(Corp\.|Corporation|Inc\.|Incorporated|Ltd\.|LLC|Co\.|Company)\b', '', name, flags=re.IGNORECASE)  # Remove legal terms and abbreviations
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
    return name.lower()

# Clean names
data1['clean_company'] = data1['company'].apply(clean_company_name)
data2['clean_company'] = data2['Company_Name'].apply(clean_company_name)

print('Sample cleaned names from Data1:', data1['clean_company'].head().tolist())
print('Sample cleaned names from Data2:', data2['clean_company'].head().tolist())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix1 = vectorizer.fit_transform(data1['clean_company'])
tfidf_matrix2 = vectorizer.transform(data2['clean_company'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Adjusted matching threshold
THRESHOLD = 0.7  # Lowered slightly for flexibility

# Matching process
matches = []
for idx, row in data1.iterrows():
    best_match_idx = cosine_sim[idx].argmax()
    best_match_score = cosine_sim[idx][best_match_idx]
    if best_match_score >= THRESHOLD:
        matched_excel_id = data2.iloc[best_match_idx]['Excel_Company_ID']
    else:
        matched_excel_id = ''
    matches.append(matched_excel_id)

data1['Matched_Excel_Company_ID'] = matches

# Identify unmatched firms
unmatched_firms = data1[data1['Matched_Excel_Company_ID'] == '']
unmatched_file = 'Unmatched_Companies.csv'
unmatched_firms.to_csv(unmatched_file, index=False)

print(f'Unmatched firms saved to {unmatched_file}')

# Save output
output_file = 'Matched_Companies.csv'
data1.to_csv(output_file, index=False)

print(f'Matching completed. Results saved to {output_file}')

# Calculate matching rate
total_matches = sum(data1['Matched_Excel_Company_ID'] != '')
matching_rate = total_matches / len(data1) * 100

print(f'Total Matches: {total_matches}')
print(f'Matching Rate: {matching_rate:.2f}%')
