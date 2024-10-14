import json
import matplotlib.pyplot as plt


train_loss = json.load(open('./logs/train_loss.json','r'))
train_miou = json.load(open('./logs/train_miou.json','r'))
test_loss = json.load(open('./logs/test_loss.json','r'))
test_miou = json.load(open('./logs/test_miou.json','r'))

fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(15,5))

X=[]
Y=[]
for x,y in train_loss:
    X.append(x)
    Y.append(y)
axes[0].plot(X,Y,label='train_loss')


X=[]
Y=[]
for x,y in test_loss:
    X.append(x)
    Y.append(y)
axes[0].plot(X,Y,label='test_loss')
axes[0].set_title('loss')
axes[0].legend()


X=[]
Y=[]
for x,y in train_miou:
    X.append(x)
    Y.append(y)
axes[1].plot(X,Y,label='train_miou')

X=[]
Y=[]
for x,y in test_miou:
    X.append(x)
    Y.append(y)
axes[1].plot(X,Y,label='test_miou')
axes[1].set_title('miou')
axes[1].legend()

plt.show()
