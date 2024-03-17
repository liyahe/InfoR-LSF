import torchtext


train, test = torchtext.datasets.AmazonReviewPolarity(root="./data/original")


amazon5 = torchtext.datasets.AmazonReviewFull(root="./data/original")


train, test = torchtext.datasets.YelpReviewPolarity(root="./data/original")

for label, text in train:
    print(type(label), text)
    break
for label, text in test:
    print(label, text)
    break
