import torch

""" labels = torch.cat([torch.arange(8) for i in range(2)], dim=0)
print("labels generated:")
print(labels)
print(labels.unsqueeze(0).size())
print(labels.unsqueeze(1).size())
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
print("labels arranging")
print(labels, labels.size())
#labels = labels.to(self.args.device)
features = torch.rand((16, 2))
print(features)
#features = F.normalize(features, dim=1)

similarity_matrix = torch.matmul(features, features.T)
print("sim mat: ",similarity_matrix.size())
# assert similarity_matrix.shape == (
#     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
# assert similarity_matrix.shape == labels.shape

# discard the main diagonal from both: labels and similarities matrix
mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.args.device)#identity matrix
print("mask: ")
print(mask, mask.size())
labels = labels[~mask].view(labels.shape[0], -1)
print("labels after masking: ")
print(labels, labels.size())
similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
print("sim matrix")
print(similarity_matrix.size())
# assert similarity_matrix.shape == labels.shape

# select and combine multiple positives
positives = similarity_matrix[labels.bool()]
print("positives: ", positives.size())
positives = positives.view(labels.shape[0], -1)
print("positives after rearranging: ", positives.size())


# select only the negatives the negatives
negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
print("negatives: ",negatives.size())

logits = torch.cat([positives, negatives], dim=1)

labels = torch.zeros(logits.shape[0], dtype=torch.long)#.to(self.args.device)

logits = logits / 2
#print("logits: ",logits)
print("logits: ",logits.size())
print("labels: ", labels, labels.size())

criterion = torch.nn.CrossEntropyLoss(reduction="none")
loss = criterion(logits, labels)
print(loss, loss.size()) """

criterion_1 = torch.nn.CosineSimilarity(dim=0)

criterion_2 = torch.nn.CosineSimilarity(dim=1)

a = torch.rand((16, 2))
b = torch.rand((16, 2))
print(a)
print(b)
sim = criterion_1(a, b)
print(sim)
sim = criterion_2(a, b)
print(sim)
