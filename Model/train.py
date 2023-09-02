import torch
import numpy as np
import torch.nn as nn
from model import CNN

train_Loader, test_Loader = torch.load("../DataLoader/train_Loader.pt"), torch.load("../DataLoader/test_Loader.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(DEVICE)
learning_rate = 0.001
OPTIMIZER = torch.optim.Adam(model.parameters(), lr = learning_rate)
LOSSFUNC = nn.CrossEntropyLoss()
EPOCH = 10

def train(model, train_Loader):
    model.train()
    training_accuracy = []
    training_loss = []

    for epoch in range(EPOCH):
        sampleCorrect = 0
        sampleTotal = 0
        losses = []
        for batch_num, data in enumerate(train_Loader):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_predicted = model(x)
            loss = LOSSFUNC(y_predicted, y.long())
            losses.append(loss.item())

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            sampleCorrect += torch.sum(torch.argmax(y_predicted, dim = 1) == y).item()
            sampleTotal += y.size(0)
        print(f"Training Accuracy for Epoch {epoch}: {np.round(sampleCorrect / sampleTotal, 4)}")
        print(f"Training Loss for Epoch {epoch}: {np.sum(losses) / len(losses)}")
        print("--------------------------------------------------")
        training_accuracy.append(np.round(sampleCorrect / sampleTotal, 4))
        training_loss.append(np.sum(losses) / len(losses))

    return (training_accuracy, training_loss)

def test(model, test_Loader):
    model.eval()

    with torch.no_grad():
        for x, y in test_Loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_predicted = model(x)
            Correct = torch.sum(torch.argmax(y_predicted, dim = 1) == y).item()
            Total = y.size(0)
    accuracy = round(Correct / Total, 4)
    print("Accuracy on Testing Set: ", accuracy)
    return accuracy

training_accuracy, training_loss = train(model, train_Loader)
test_accuracy = test(model, test_Loader)

torch.save(model.state_dict(), "../model/model.pth")