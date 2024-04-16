
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import os.path as osp
import time
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
# parser.add_argument('--dataset', type=str, default='Cora',
#                     choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=35)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
    #                   split_labels=True, add_negative_train_samples=False),
])

def string_to_id(paragraphs):
    dict_ = {}
    s = set()
    dict2 = {}
    tokenized_paragraphs = [paragraph.split() for paragraph in paragraphs]
    model = Word2Vec(tokenized_paragraphs, vector_size=100, window=25, min_count=1, workers=4, sg = 0, epochs = 10)

    for paragraph in paragraphs:
        for word in paragraph.split():
            s.update([word])
    sorted_words = sorted(list(s))
    s = set(sorted_words)
    dict_ = {word: i for i, word in enumerate(sorted_words)}
    dict2 = {dict_[word]: model.wv.get_vector(word) for i, word in enumerate(sorted_words)}
   
    return dict_, dict2

def construct_graph(paragraphs):
    graph = nx.Graph()
    for paragraph in paragraphs:
        words = paragraph
        graph.add_nodes_from(words)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):

                if graph.has_edge(words[i], words[j]):
                    graph[words[i]][words[j]]['weight'] += 1
                    graph[words[j]][words[i]]['weight'] += 1
                else:   
                    graph.add_edge(words[i], words[j], weight=1)
    return graph

def normalize_weights(graph):
    for u, v, data in graph.edges(data=True):
        data['weight'] /= graph.degree[u] * graph.degree[v]

# read data 
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
                # paragraphs.append(' '.join(x[1:]))
                paragraphs.append(' '.join(x[0:]))

    return paragraphs

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

    sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_paragraphs

folder_path = '/users/anhld/BiasInRSSE/CROSSREC/D2'
paragraphs = read_files(folder_path)

list_q = [
    'io.reactivex.rxjava2:rxandroid',
    'com.squareup.leakcanary:leakcanary-android',
    'com.github.bumptech.glide:glide',
    'com.squareup.okhttp3:logging-interceptor',
    'com.squareup.leakcanary:leakcanary-android-no-op',
]
# sorted_paragraphs_1 = retrieval1(paragraphs, list_q)
# count = 0
# tokenized_paragraphs = []

# for paragraph, score in sorted_paragraphs_1.items():
#     # print(f"Project {count}:", paragraph)
#     # print("Cosine Score:", score)
#     print(count)
#     print(paragraph)

#     if count==7:
#         count += 1
#         continue
#     # tokenized_paragraphs.append(paragraph.split())
#     tokenized_paragraphs.append(paragraph)

#     count += 1
#     if count == 16:
#         break

dict_, dict2 = string_to_id(paragraphs)
# print(dict_)
# print(dict2)

values_tensor = torch.tensor(list(dict2.values()))  # Convert values to a tensor
# Combine keys and values tensors into a single tensor

new_paragraphs = []
for paragraph in paragraphs:
    words = paragraph.split()
    x = []
    for word in words:
        x.append(dict_[word])
    new_paragraphs.append(x)

graph = construct_graph(new_paragraphs)
# normalize_weights(graph)
edges = np.array(list(graph.edges))
# edges = edges.T
print(graph)
edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()

# np.random.shuffle(edges)
half_edge = edges.shape[0] // 2
print(half_edge)
random_pos_edge_indices = edges[:half_edge, :]
pos_edge_label_index = torch.tensor(random_pos_edge_indices, dtype=torch.long).t().contiguous()
random_neg_edge_indices = edges[half_edge:, :]
neg_edge_label_index = torch.tensor(random_neg_edge_indices, dtype=torch.long).t().contiguous()
print(pos_edge_label_index.shape)
train_data = Data(x=values_tensor, edge_index=edges_tensor, y=None, pos_edge_label_index = pos_edge_label_index, neg_edge_label_index=neg_edge_label_index).to(device)

print(train_data)
# Data(x=[2708, 1433], edge_index=[2, 8976], pos_edge_label_index=[2, 4488])
# print(train_data.pos_edge_label.sum())
# print(train_data.pos_edge_label_index.max())

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# in_channels, out_channels = dataset.num_features, 16

in_channels, out_channels = 100, 100


if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    auc, ap = test(train_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


# def retrieval(paragraphs, list_query, model, data, dict_):
#     top_similar_paragraph = {}
#     z = model.encode(data.x, data.edge_index)
#     #  shape [61,300]

#     for paragraph in paragraphs:
#         cosine_score = 0
#         for word in paragraph.split():
#             if word in list_query:
#                 continue
#             if word not in dict_.keys():
#                 continue 
#             # if word in model.wv.key_to_index.keys():  # Check if word is in vocabulary
#             for query in list_query:
#                 if query not in dict_.keys():
#                     break 
#                 word_emb = z[dict_[word]]
#                 query_emb = z[dict_[query]]
#                 word_emb_cpu = word_emb.cpu().detach().numpy().reshape(1,-1)
#                 query_emb_cpu = query_emb.cpu().detach().numpy().reshape(1,-1)

#                 # Calculating similarity
#                 similarity = cosine_similarity(word_emb_cpu, query_emb_cpu)
#                 # print(similarity[0].item())
#                 cosine_score += similarity
#             # else: 
#             #     continue
#             top_similar_paragraph[word] = cosine_score
#             cosine_score = 0

#     # Sort the dictionary by cosine_score in descending order
#     sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1], reverse=True))
#     # sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1][0], reverse=True))
#     return sorted_paragraphs


def retrieval(dict_, list_query, model, data):
    top_similar_paragraph = {}
    z = model.encode(data.x, data.edge_index)
    #  shape [61,300]

    for word in dict_.keys():
        cosine_score = 0
        if 'git://github.com' in  word:
            continue 
        if word in list_query:
            continue
        if word not in dict_.keys():
            continue 
        # if word in model.wv.key_to_index.keys():  # Check if word is in vocabulary
        for query in list_query:
            if query not in dict_.keys():
                break 
            word_emb = z[dict_[word]]
            query_emb = z[dict_[query]]
            word_emb_cpu = word_emb.cpu().detach().numpy().reshape(1,-1)
            query_emb_cpu = query_emb.cpu().detach().numpy().reshape(1,-1)

            # Calculating similarity
            similarity = cosine_similarity(word_emb_cpu, query_emb_cpu)
            # print(similarity[0].item())
            cosine_score += similarity
        # else: 
        #     continue
        top_similar_paragraph[word] = cosine_score
        cosine_score = 0

# Sort the dictionary by cosine_score in descending order
    sorted_paragraphs = dict(sorted(top_similar_paragraph.items(), key=lambda x: x[1], reverse=True))
    return sorted_paragraphs


sorted_paragraphs = retrieval(dict_, list_q, model, train_data)
count = 0
for paragraph, score in sorted_paragraphs.items():
    print(f"Project {count}:", paragraph)
    # print("Cosine Score:", score)
    count += 1
    if count == 51:
        break
