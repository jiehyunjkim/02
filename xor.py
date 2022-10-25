import torch, torchmetrics
from torch import nn, utils
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

class XORDataset(Dataset):
    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label



class XOR_lightning(pl.LightningModule):
    def __init__(self, model, learning_rate: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        if model is None:
            self.model = SimpleClassifier()
        else:
            self.model = model
        self.loss_module = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()


    def training_step(self, batch, batch_idx):
        x, y = batch
        # Determine prediction of model on dev set
        preds = self.model(x)
        preds = preds.squeeze(dim=1)
        loss = self.loss_module(preds, y.float())
        preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
        pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1
        self.train_acc(pred_labels, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Determine prediction of model on dev set
        preds = self.model(x)
        preds = preds.squeeze(dim=1)
        loss = self.loss_module(preds, y.float())
        preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
        pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1
        self.valid_acc(pred_labels, y)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

'''    
class dataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
'''

@torch.no_grad() # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
def visualize_classification(model, data, label):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(4,4), dpi=120)
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    # Let's make use of a lot of operations we have learned above
    model.to(device)
    c0 = torch.Tensor(to_rgba("C0")).to(device)
    c1 = torch.Tensor(to_rgba("C1")).to(device)
    x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    output_image = (1 - preds) * c0[None,None] + preds * c1[None,None]  # Specifying "None" in a dimension creates a new one
    output_image = output_image.cpu().numpy()  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig


def xor_main():
    #set datasets
    train_dataset = XORDataset(size=200)
    valid_dataset = XORDataset(size=200)

    train_loader = utils.data.DataLoader(train_dataset)
    valid_loader = utils.data.DataLoader(valid_dataset)

    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    xor = XOR_lightning(model)

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=10, accelerator='cpu', devices=1)
    trainer.fit(model=xor, train_dataloaders=train_loader, val_dataloaders=valid_loader) 

    _ = visualize_classification(model, valid_dataset.data, valid_dataset.label)
    plt.show()


if __name__ == "__main__":
    xor_main()
    