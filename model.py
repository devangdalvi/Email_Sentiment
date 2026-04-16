import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SentimentGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_classes=3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels // 2)

        self.fc = torch.nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        return self.fc(x)