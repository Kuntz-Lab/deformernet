import torch
import torch.optim as optim
import torch.nn.functional as F
from dnn_architecture import Net
from dataset_loader import ShapeServoDataset
from torch.utils.tensorboard import SummaryWriter

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    num_batch = 0
    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
        data = sample["input"].to(device)
        target = sample["target"].to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss/num_batch))  
    # writer.add_scalar('training loss', train_loss/num_batch, epoch)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample["input"].to(device)
            target = sample["target"].to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # writer.add_scalar('test loss',test_loss, epoch)

    # print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))


if __name__ == "__main__":
    
    # writer = SummaryWriter('runs/VFH_PCA_method_with_relu_2')
    
    torch.manual_seed(2021)
    device = torch.device("cuda")

    # train_dataset = ShapeServoDataset(percentage = .8) 
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # test_dataset = ShapeServoDataset(percentage = .2)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)


    train_len = 2700
    test_len = 300
    total_len = train_len + test_len

    dataset = ShapeServoDataset(percentage = 1.0)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)


    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    for epoch in range(1, 201):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        if epoch % 10 == 0:
            test(model, device, test_loader, epoch)


    # torch.save(model.state_dict(), "/home/baothach/shape_servo_data/weights/VFH_PCA_with_relu_2")

