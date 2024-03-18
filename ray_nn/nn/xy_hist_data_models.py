import math
import torch
from torch import optim, nn, utils
import lightning as L

from ray_tools.simulation.torch_datasets import RayDataset


# define the LightningModule
class MetrixXYHistSurrogate(L.LightningModule):
    def __init__(self, layer_size:int=2, blow:float=100., shrink_factor:str='log', learning_rate:float=0.001, gpus:int=0, optimizer:str='adam', autoencoder_checkpoint:str='../../data/ae_lrelu_epoch=55-step=52527.ckpt'):
        super(MetrixXYHistSurrogate, self).__init__()
        self.save_hyperparameters()

        self.net = self.create_sequential(35, 100, self.hparams.layer_size, blow=self.hparams.blow, shrink_factor=self.hparams.shrink_factor)
        print(self.net)

    def create_sequential(self, input_length, output_length, layer_size, blow=0, shrink_factor="log"):
        layers = [input_length]
        blow_disabled = blow == 1 or blow == 0
        if not blow_disabled:
            layers.append(input_length*blow)

        if shrink_factor == "log":
            add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length,10), steps=layer_size+2-len(layers), base=10).long()
            # make sure the last element is correct, even though rounding
            add_layers[-1] = output_length
        elif shrink_factor == "lin":
            add_layers = torch.linspace(layers[-1], output_length, steps=layer_size+2-len(layers)).long()
        else:
            shrink_factor = float(shrink_factor)
            new_length = layer_size+1-len(layers)
            add_layers = (torch.ones(new_length)*layers[-1] * ((torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))).long()
            layers = torch.cat((torch.tensor([input_length]), add_layers))
            layers = torch.cat((layers, torch.tensor([output_length])))
    
        if not blow_disabled:
            layers = torch.tensor([layers[0]])
            layers = torch.cat((layers, add_layers))
        else:
           layers = add_layers

        nn_layers = []
        for i in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[i].item(), layers[i+1].item()))
            if not i == len(layers)-2:
                nn_layers.append(nn.ReLU())
                #nn_layers.append(nn.BatchNorm1d(layers[i+1].item()))
        return nn.Sequential(*nn_layers)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = torch.cat(list(batch['1e5/params'].values())).float().unsqueeze(0).unsqueeze(0)
        y = batch['1e5/histogram'].float()
        if 0 in y.shape:
            y = torch.zeros(torch.Size([y.size()[0], y.size()[1], 1, y.size()[-1]]))
        y = y.flatten(start_dim=1)
        y_hat = self.net(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = MetrixXYHistSurrogate()

dataset = RayDataset(h5_files=['datasets/metrix_simulation/ray_emergency_surrogate/50+50_data_raw_0.h5'],
                     sub_groups=['1e5/params',
                                 '1e5/histogram'])
train_loader = utils.data.DataLoader(dataset)

trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_loader)
