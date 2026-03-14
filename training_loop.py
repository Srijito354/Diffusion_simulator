import torch
import torch.nn as nn
from custom_dataset import Data
from Scheduler import Noise_scheduler
from My_model import Diffusion_model
from torch.utils.data import Dataset, DataLoader

data_path = "Datashape.tsv"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Diffusion_model()
#model.load_state_dict(torch.load("trained_model13.pt", map_location = device)) # Best, so far. 12th March, 19:25 IST.
#model.load_state_dict(torch.load("trained_model14.pt", map_location = device))
model.load_state_dict(torch.load("checkpoints2/trained_model600.pt", map_location = device))

scheduler = Noise_scheduler(device = device)
dataset = Data(data_path)

def train(model, scheduler, dataset, epochs = 5000, batch_size = 4, lr = 3e-4):
    model.train()

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr = lr, weight_decay = 1e-5
    )

    '''
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = epochs,  eta_min = 1e-5
    )
    '''

    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    criterion = nn.MSELoss() # using mean squared error, since we are dealing with points.

    model.to(device)

    for epoch in range(epochs):
        total_loss = []

        for batch in dataloader:
            coordinates = batch["Coordinate"].to(device)
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)

            #x0 = coordinates.unsqueeze(1)
            x0 = coordinates

            t = torch.randint(0, scheduler.T, (x0.shape[0],)).to(device)

            # adding noise, using the scheduler (going forwards, baby!)
            x_t, noise = scheduler.add_noise(x0, t)

            x_t = x_t.to(device)
            noise = noise.to(device)

            # predict noise.
            predicted_noise = model(x_t, t, input_ids, attention_mask)

            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler_lr.step()

            total_loss.append(loss.item())

        if (epoch + 1)%10 == 0:
            torch.save(model.state_dict(), f"checkpoints3/trained_model{epoch+1}.pt")        

        avg_loss = sum(total_loss)/len(total_loss)

        scheduler_lr.step(avg_loss)

        print(f"{epoch + 1}st epoch | with average loss: {avg_loss:.4f}")
    
    #torch.save(model.state_dict(), "trained_model1.pt")

train(model, scheduler, dataset)