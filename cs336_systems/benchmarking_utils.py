import torch
from timeit import timeit
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.data_utils import get_model_config, BenchmarkConfig
from functools import wraps
import numpy as np
import pandas as pd
from typing import Tuple


def generate_random_data(config: BenchmarkConfig) -> torch.Tensor:
    """Generate random input data based on config."""
    return torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.context_length),
        device=config.to_device(),
    )


def get_model(config: BenchmarkConfig) -> BasicsTransformerLM:
    """Create and configure model based on config."""
    model = BasicsTransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.rope_theta,
    )
    model.to(config.to_device())
    return model


def cnt_time(f, n=5):
    @wraps(f)
    def wrap(*args, **kwargs) -> tuple[float, float]:
        assert n > 1
        times = np.array(
            [timeit(lambda: f(*args, **kwargs), number=1) for _ in range(n)]
        )
        return times.mean().item(), times.std().item()

    return wrap


def run_model(
    model: BasicsTransformerLM, input_data: torch.Tensor, backward: bool = False
):
    output = model(input_data)
    if backward:
        fake_loss = output.mean()
        fake_loss.backward()
    # Get device from any model parameter (they're all on the same device)
    device = next(model.parameters()).device.type
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def benchmark(
    model: BasicsTransformerLM,
    input_data: torch.Tensor,
    warm_up_steps: int,
    n_steps: int,
    backward: bool = False,
) -> Tuple[float, float]:
    """Run benchmark with warmup and timing."""
    if warm_up_steps:
      _ = cnt_time(run_model, warm_up_steps)(model, input_data, backward)
    clock_time = cnt_time(run_model, n_steps)(model, input_data, backward)
    return clock_time


def task_1(
    config=BenchmarkConfig(),
    out_file="cs336_systems/data/task1_v2.md",
):
    """Run benchmarks across model sizes, devices, and other parameters."""
    from itertools import product

    # Define all dimensions to test
    test_configs = {
        "model_size": ("small", "medium"),
        "device": ("cpu", "mps"),  # Test both CPU and MPS
        "backward": (False, True),
        "warmup_steps": (0, 5),
    }

    data = []

    # Generate input data once - it will be moved to the right device as needed
    mock_input = generate_random_data(config)

    # Generate all combinations of test parameters
    for ms, dev, bd, ws in product(
        test_configs["model_size"],
        test_configs["device"],
        test_configs["backward"],
        test_configs["warmup_steps"],
    ):
        # Create config for this run
        run_config = BenchmarkConfig(
            **{**vars(config), "model_size": ms, "device": dev, "warm_up_steps": ws}
        )
        print(run_config)

        # Create model on the target device
        model = get_model(run_config)
        # Move input to the current device
        run_input = mock_input.to(run_config.to_device())

        try:
            mean, std = benchmark(
                model, run_input, warm_up_steps=ws, n_steps=10, backward=bd
            )
            status = "ok"
        except Exception as e:
            mean, std = float("nan"), float("nan")
            status = str(e)

        data.append(
            {
                "model_size": ms,
                "device": dev,
                "+backward": bd,
                "warmup_steps": ws,
                "mean": mean,
                "std": std,
                "status": status,
            }
        )

    df = pd.DataFrame(data)
    # Sort by model size, device, and other parameters for better readability
    df = df.sort_values(["model_size", "device", "+backward", "warmup_steps"])

    df.to_markdown(out_file, tablefmt="grid", floatfmt=".6f")
    return df


if __name__ == "__main__":
    task_1()
