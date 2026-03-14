import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Diffusion_model(nn.Module):
    def __init__(self, num_of_timesteps = 1000, time_embed_dim = 512, hidden_dim = 1024):
        super().__init__()

        #self.bert = AutoModel.from_pretrained("bert-based-uncased", token = os.getenv("HF_TOKEN"))
        self.bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only = True)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.time_embedding = nn.Embedding(num_of_timesteps, time_embed_dim)

        # Creating conditional (text + time) projection.
        self.condition_proj = nn.Sequential(
            nn.Linear(768 + time_embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )

        # Creating the denoising network. Takes in the [noisy coordinates(of size (2, )) + condition(of size(256,))] as input.
        self.point_mlp = nn.Sequential(
            nn.Linear(2 + hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, noisy_points, start_timestep, input_ids, attention_mask):

        # creating text encoding.
        bert_out = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls_vector = bert_out.last_hidden_state[:, 0, :] # the [CLS] vector is a summary of all the vectors of the sentence, produced by BERT.

        # creating the time encoding.
        t_embed = self.time_embedding(start_timestep)

        # creating time + text prompt condition.
        condition = torch.cat([cls_vector, t_embed], dim = -1) # We choose the last dimension, since the embeddings are shaped like (batch_size, 1, embedding_dim). We are concatenating on the embedding dimension.
        condition = self.condition_proj(condition)

        N_shape = noisy_points.shape[1]
        condition = condition.unsqueeze(1).expand(-1, N_shape, -1)

        x = torch.cat([noisy_points, condition], dim = -1)

        predicted_noise = self.point_mlp(x)

        return predicted_noise