from typing import List, Callable, Union
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int,
        activation_func: List[Callable[[torch.Tensor], torch.Tensor]],
        with_stdv: bool = False,
    ):
        super().__init__()

        self.with_stdv = with_stdv

        self.mlp_without_output = nn.ModuleList()
        self.mlp_without_output.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        self.mlp_without_output.append(activation_func[0])

        for i in range(1, len(hidden_layer_sizes) - 1):
            self.mlp_without_output.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            )
            self.mlp_without_output.append(activation_func[i])

        self.mean_output = nn.Linear(hidden_layer_sizes[-1], output_size)
        if with_stdv:
            self.stdv_output = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        x = x.float()
        for layer in self.mlp_without_output:
            x = layer(x)

        action_means = torch.nn.functional.softplus(self.mean_output(x))

        if self.with_stdv:
            action_stdv = torch.nn.functional.softplus(self.stdv_output(x))
            return action_means, action_stdv

        return action_means


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    mlp = MLP(
        4, [256, 256, 256], 2, [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Identity()], True
    ).to(device)
    print(mlp)

    test = torch.tensor([1, 2, 3, 4], device=device)
    torch.save(mlp, "./policies/DroneWorld/REINFORCE/hoverPolicy.pth")
    model = torch.load("./policies/DroneWorld/REINFORCE/hoverPolicy.pth")
    print(model(test))
    print(mlp(test))
