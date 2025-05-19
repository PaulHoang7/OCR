import timm
import torch.nn as nn
import torch


# class CRNN(nn.Module):
#     def __init__(
#         self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
#     ):
#         super(CRNN, self).__init__()

#         backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
#         modules = list(backbone.children())[:-2]
#         modules.append(nn.AdaptiveAvgPool2d((1, None)))
#         self.backbone = nn.Sequential(*modules)

#         # Unfreeze the last few layers
#         for parameter in self.backbone[-unfreeze_layers:].parameters():
#             parameter.requires_grad = True

#         self.mapSeq = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout))

#         self.gru = nn.GRU(
#             512,
#             hidden_size,
#             n_layers,
#             bidirectional=True,
#             batch_first=True,
#             dropout=dropout if n_layers > 1 else 0,
#         )
#         self.layer_norm = nn.LayerNorm(hidden_size * 2)

#         self.out = nn.Sequential(
#             nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
#         )

#     @torch.autocast(device_type="cuda")
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.permute(0, 3, 1, 2)
#         x = x.view(x.size(0), x.size(1), -1)  # Flatten the feature map
#         x = self.mapSeq(x)
#         x, _ = self.gru(x)
#         x = self.layer_norm(x)
#         x = self.out(x)
#         x = x.permute(1, 0, 2)  # Based on CTC

#         return x

# class CRNN(nn.Module):
#     def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2):
#         super(CRNN, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Changed

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(5, 5),  # Changed

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),  # Changed

#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, None))
#         )

#         self.mapSeq = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         self.gru = nn.GRU(
#             512,
#             hidden_size,
#             n_layers,
#             bidirectional=True,
#             batch_first=True,
#             dropout=dropout if n_layers > 1 else 0
#         )
#         self.layer_norm = nn.LayerNorm(hidden_size * 2)

#         self.out = nn.Sequential(
#             nn.Linear(hidden_size * 2, vocab_size),
#             nn.LogSoftmax(dim=2)
#         )

#     def forward(self, x):
#         x = self.cnn(x)  # (B, C=512, H=1, W=14)
#         x = x.squeeze(2)  # (B, 512, W=14)
#         x = x.permute(0, 2, 1)  # (B, W=14, 512) —> seq_len = 14

#         x = self.mapSeq(x)      # (B, 14, 512)
#         x, _ = self.gru(x)      # (B, 14, hidden*2)
#         x = self.layer_norm(x)
#         x = self.out(x)         # (B, 14, vocab_size)
#         x = x.permute(1, 0, 2)  # (14, B, vocab_size) —> CTC format
#         return x



class CRNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, 21)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout))

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the feature map
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # Based on CTC

        return x