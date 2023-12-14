import torch

class BrainTumorModel(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.5),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.5),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=64,
                            kernel_size=(3, 3),
                            stride=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2)),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64*109*109,
                            out_features = output_shape)
        )
        

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        #print(f"Output shape of conv_block_1: {x.shape}")

        x = self.classifier(x)
        #print(f"Output shape of classifier: {x.shape}")

        return x