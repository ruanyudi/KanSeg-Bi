import json
import matplotlib.pyplot as plt


train_loss = json.load(open("./logs/train_loss.json", "r"))
train_miou = json.load(open("./logs/train_miou.json", "r"))
test_loss = json.load(open("./logs/test_loss.json", "r"))
test_miou = json.load(open("./logs/test_miou.json", "r"))
test_f1score = json.load(open("./logs/test_f1scores.json", "r"))
test_f1score = sorted(test_f1score, key=lambda x: x[0])
test_precision = json.load(open("./logs/test_precisions.json", "r"))
test_precision = sorted(test_precision, key=lambda x: x[0])
test_recall = json.load(open("./logs/test_recalls.json", "r"))
test_recall = sorted(test_recall, key=lambda x: x[0])

train_f1score = json.load(open("./logs/train_f1scores.json", "r"))
train_f1score = sorted(train_f1score, key=lambda x: x[0])
train_precision = json.load(open("./logs/train_precisions.json", "r"))
train_precision = sorted(train_precision, key=lambda x: x[0])
train_recall = json.load(open("./logs/train_recalls.json", "r"))
train_recall = sorted(train_recall, key=lambda x: x[0])
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 5))

X = []
Y = []
for x, y in train_loss:
    X.append(x)
    Y.append(y)
axes[0, 0].plot(X, Y, label="train_loss")


X = []
Y = []
for x, y in test_loss:
    X.append(x)
    Y.append(y)
axes[0, 0].plot(X, Y, label="test_loss")
axes[0, 0].set_title("loss")
axes[0, 0].legend()


X = []
Y = []
for x, y in train_miou:
    X.append(x)
    Y.append(y)
axes[0, 1].plot(X, Y, label="train_miou")

X = []
Y = []
for x, y in test_miou:
    X.append(x)
    Y.append(y)
axes[0, 1].plot(X, Y, label="test_miou")
# axes[1].set_title('miou')
axes[0, 1].legend()


X = []
Y = []
for x, y in test_f1score:
    X.append(x)
    Y.append(y)
axes[1, 0].plot(X, Y, label="test_f1score")
# axes[1].set_title('miou')
axes[1, 0].legend()


X = []
Y = []
for x, y in test_precision:
    X.append(x)
    Y.append(y)
axes[1, 0].plot(X, Y, label="test_precision")
# axes[1].set_title('miou')
axes[1, 0].legend()


X = []
Y = []
for x, y in test_recall:
    X.append(x)
    Y.append(y)
axes[1, 0].plot(X, Y, label="test_recall")
# axes[1].set_title('miou')
axes[1, 0].legend()


X = []
Y = []
for x, y in train_f1score:
    X.append(x)
    Y.append(y)
axes[1, 1].plot(X, Y, label="train_f1score")
# axes[1].set_title('miou')
axes[1, 1].legend()


X = []
Y = []
for x, y in train_precision:
    X.append(x)
    Y.append(y)
axes[1, 1].plot(X, Y, label="train_precision")
# axes[1].set_title('miou')
axes[1, 1].legend()


X = []
Y = []
for x, y in train_recall:
    X.append(x)
    Y.append(y)
axes[1, 1].plot(X, Y, label="train_recall")
# axes[1].set_title('miou')
axes[1, 1].legend()
plt.show()
