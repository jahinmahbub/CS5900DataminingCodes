# -------------------------------------------------------------------------
# AUTHOR: Jahin Mahbub
# FILENAME: similarity.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 20 minutes
# -----------------------------------------------------------*/

# Importing standard Python libraries
import csv
import math

# Step 1: Read documents from the CSV file
documents = []  # Store all document texts
document_ids = []  # Store corresponding document IDs

with open('cleaned_documents.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    for row in reader:
        document_ids.append(int(row[0]))  # Store the document ID
        documents.append(row[1])  # Store the document text

# Step 2: Build the vocabulary (set of unique words in all documents)
vocabulary = set()
for doc in documents:
    words = doc.split()  # Tokenizing by space
    vocabulary.update(words)

vocabulary = sorted(vocabulary)  # Sort for consistent word ordering

# Step 3: Create the binary Document-Term Matrix
docTermMatrix = []

for doc in documents:
    words = set(doc.split())  # Unique words in the document
    vector = [1 if word in words else 0 for word in vocabulary]  # Binary encoding
    docTermMatrix.append(vector)

# Step 4: Compute pairwise cosine similarity
def cosine_similarity(vec1, vec2):
    """ Compute the cosine similarity between two binary vectors """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Avoid division by zero
    return dot_product / (magnitude1 * magnitude2)

# Step 5: Find the two most similar documents
max_similarity = 0
most_similar_docs = (-1, -1)

for i in range(len(docTermMatrix)):
    for j in range(i + 1, len(docTermMatrix)):  # Avoid redundant comparisons
        similarity = cosine_similarity(docTermMatrix[i], docTermMatrix[j])
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_docs = (document_ids[i], document_ids[j])  # Get actual document IDs

# Step 6: Print the result
print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity:.4f}")