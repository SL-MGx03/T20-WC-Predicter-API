import torch
from torch import nn

class Win_Pred_Model(nn.Module):
    def __init__(self, n_teams, n_cities, team_emb_dim=8, city_emb_dim=6, hidden_units=16, dropout=0.2):
        super().__init__()

        self.team_emb = nn.Embedding(n_teams, team_emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)

        # numeric features: runs + wkts = 2
        in_features = team_emb_dim + team_emb_dim + city_emb_dim + 2

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, team_a_id, team_b_id, city_id, x_num):
        # team_a_id, team_b_id, city_id: (batch,)
        # x_num: (batch, 2)
        ea = self.team_emb(team_a_id)
        eb = self.team_emb(team_b_id)
        ec = self.city_emb(city_id)

        x = torch.cat([ea, eb, ec, x_num], dim=1)
        return self.net(x).squeeze(1)  # logits
        