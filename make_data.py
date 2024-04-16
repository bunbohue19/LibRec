
    
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

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

folder_path = '/users/anhld/BiasInRSSE/CROSSREC/D2'
paragraphs = read_files(folder_path)

def retrieval1(paragraphs, list_query):
    top_similar_paragraph = {}

    corpus = paragraphs
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    for paragraph in paragraphs:
        x1 = vectorizer.transform([paragraph])  
        x2 = vectorizer.transform([' '.join(list_query)])
        cosine_score = cosine_similarity(x1, x2)
        top_similar_paragraph[paragraph] = cosine_score

    # Sort the dictionary by cosine_score in descending order
    sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_paragraphs

# list_q = [
# 'com.squareup.okhttp3:okhttp'
# 'com.android.support:recyclerview-v7'
# 'com.jakewharton:butterknife-compiler'
# 'com.onesignal:OneSignal'
# 'com.android.support:appcompat-v7'
# 'com.google.code.gson:gson'
# 'com.android.support:customtabs'
# ]
list_q = [
    'io.reactivex.rxjava2:rxandroid',
    'com.squareup.leakcanary:leakcanary-android',
    'com.github.bumptech.glide:glide',
    'com.squareup.okhttp3:logging-interceptor',
    'com.squareup.leakcanary:leakcanary-android-no-op',
]
sorted_paragraphs_1 = retrieval1(paragraphs, list_q)
count = 0
tokenized_paragraphs = []

for paragraph, score in sorted_paragraphs_1.items():
    # print(f"Project {count}:", paragraph)
    # print("Cosine Score:", score)
    if count==0:
        count += 1
        continue
    # tokenized_paragraphs.append(paragraph.split())
    tokenized_paragraphs.append(paragraph)

    count += 1
    if count == 16:
        break
# sorted_paragraphs = retrieval(paragraphs, list_q, model)

# count = 0
# for paragraph, score in sorted_paragraphs.items():
#     print(f"Project {count}:", paragraph)
#     # print("Cosine Score:", score)
#     print()
#     count += 1
#     if count == 16:
#         break



# # # Tokenize each paragraph into words
# tokenized_paragraphs = [paragraph.split() for paragraph in paragraphs]

# # Build Word2Vec model
# model = Word2Vec(tokenized_paragraphs, vector_size=300, window=25, min_count=1, workers=4, sg = 0, epochs = 10)

# # # Example usage: getting similarity between words
# # similarity = model.wv.similarity('#DEP#javax.enterprise:cdi-api', '#DEP#org.springframework:spring-context')
# # print("Similarity between 'word1' and 'word2':", similarity)



# def retrieval(paragraphs, list_query, model):
#     top_similar_paragraph = {}

#     for paragraph in paragraphs:
#         cosine_score = 0
#         for word in paragraph.split():
#             if word in list_query:
#                 continue
#             if word in model.wv.key_to_index.keys():  # Check if word is in vocabulary
#                 for query in list_query:
#                     similarity = model.wv.similarity(query, word)
#                     cosine_score += similarity
#             else: 
#                 continue
#             top_similar_paragraph[word] = cosine_score
#             cosine_score = 0

#     # Sort the dictionary by cosine_score in descending order
#     sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1], reverse=True))
    
#     return sorted_paragraphs
# sorted_paragraphs = retrieval(paragraphs, list_q, model)
# count = 0
# for paragraph, score in sorted_paragraphs.items():
#     print(f"Project {count}:", paragraph)
#     # print("Cosine Score:", score)
#     print()
#     count += 1
#     if count == 31:
#         break




import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

def string_to_id(paragraphs):
    dict_ = {}
    i = 0
    s = set()

    for paragraph in paragraphs:
        for word in paragraph.split():
            s.update([word])

    sorted_words = sorted(list(s))

  
    s = set(sorted_words)

    dict_ = {word: i for i, word in enumerate(sorted_words)}
    return dict_


# Step 1: Constructing the Graph
def construct_graph(paragraphs):
    graph = nx.Graph()
    for paragraph in paragraphs:
        words = paragraph
        graph.add_nodes_from(words)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if graph.has_edge(words[i], words[j]):
                    graph[words[i]][words[j]]['weight'] += 1
                else:   
                    graph.add_edge(words[i], words[j], weight=1)
    return graph

# Step 2: Normalize Edge Weights
def normalize_weights(graph):
    for u, v, data in graph.edges(data=True):
        data['weight'] /= graph.degree[u] * graph.degree[v]

# Step 3: Define GNN Architecture
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Step 4: Training
def train(model, data, train_loader, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_loader], data.y[train_loader])
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 5: Node Embeddings
def get_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
    return out

# Step 6: Evaluation
# Your evaluation code here

# Example usage
dict_ = string_to_id(tokenized_paragraphs)
print(dict_)
new_paragraphs = []
for paragraph in tokenized_paragraphs:
    words = paragraph.split()
    x = []
    for word in words:
        x.append(dict_[word])
    new_paragraphs.append(x)

graph = construct_graph(new_paragraphs)
normalize_weights(graph)
edges = np.array(list(graph.edges))
edges = edges.T

edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()

print(torch.tensor(list(dict_.values())).shape)

print(edges_tensor.shape)

data = Data(x=torch.tensor(list(dict_.values())), edge_index=edges_tensor, y=None)
model = GNN(input_dim=300, hidden_dim=64, output_dim=300)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_loader = data
train(model, data, train_loader, optimizer, criterion)
embeddings = get_node_embeddings(model, data)
