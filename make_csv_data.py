from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd 
def read_files(folder_path):
    paragraphs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        x = [] 
        paragraph = []
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip().split()
                    x.append(line[1].replace('#DEP#',''))                    
                paragraphs.append(' '.join(x[1:]))
                # paragraphs.append(' '.join(x[0:]))

    return paragraphs

def retrieval(paragraphs, list_query, vectorizer):
    top_similar_paragraph = {}
    x2 = vectorizer.transform([' '.join(list_query)])

    for paragraph in paragraphs:
        if paragraph == ' '.join(list_query):
            continue
        x1 = vectorizer.transform([paragraph])
        cosine_score = cosine_similarity(x1, x2)
        if cosine_score[0][0] >= 0.7:
            top_similar_paragraph[paragraph] = cosine_score[0][0]

    sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1], reverse=True))
    return sorted_paragraphs


folder_path = '/users/anhld/BiasInRSSE/CROSSREC/D2'
paragraphs = read_files(folder_path)

corpus = paragraphs
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)

final_text = []
final_labels = []
count1 = 0
for paragraph1 in paragraphs:
    list_q = paragraph1.split()
    sorted_paragraphs_1 = retrieval(paragraphs, list_q, vectorizer)
    count = 0
    ans = '<s>'
    for paragraph, score in sorted_paragraphs_1.items():
        # print(f"Project {count}:", paragraph)
        # print("Cosine Score:", score)
        ans += ' '
        ans += paragraph.strip()
        ans += ' </s>'
        final_text.append(ans)
        count += 1
        if count == 5:
            break
    final_labels.append(paragraph1)
    print(count1)
    count1 += 1
    


import csv 
data = list(zip(final_text, final_labels))
file_name = "output.csv"

# Write the data to a CSV file
with open(file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['text', 'labels'])
    # Write each pair of text and label
    writer.writerows(data)

print("CSV file written successfully.")


# list_q = [
#     'io.reactivex.rxjava2:rxandroid',
#     'com.squareup.leakcanary:leakcanary-android',
#     'com.github.bumptech.glide:glide',
#     'com.squareup.okhttp3:logging-interceptor',
#     'com.squareup.leakcanary:leakcanary-android-no-op',
# ]
# sorted_paragraphs_1 = retrieval1(paragraphs, list_q)
# count = 0
# tokenized_paragraphs = []

# for paragraph, score in sorted_paragraphs_1.items():
#     # print(f"Project {count}:", paragraph)
#     # print("Cosine Score:", score)
#     if count==0:
#         count += 1
#         continue
#     # tokenized_paragraphs.append(paragraph.split())
#     tokenized_paragraphs.append(paragraph)

#     count += 1
#     if count == 16:
#         break