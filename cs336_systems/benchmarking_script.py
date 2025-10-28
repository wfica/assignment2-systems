from argparse import ArgumentParser
from cs336_systems.benchmarking_utils import (
    benchmark,
    get_model,
    generate_random_data,
)
from cs336_systems.data_utils import BenchmarkConfig
import torch


def main():
    # Define all configuration options
    arg_parser = ArgumentParser(description="Benchmark transformer model configurations")
    
    # Model configuration
    model_group = arg_parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_size", type=str, default="",
                          help="Optional model size preset: small, medium, large, xl, 2.7B")
    model_group.add_argument("--vocab_size", type=int, default=10_000)
    model_group.add_argument("--context_length", type=int, default=256)
    model_group.add_argument("--d_model", type=int, default=768)
    model_group.add_argument("--num_layers", type=int, default=12)
    model_group.add_argument("--num_heads", type=int, default=12)
    model_group.add_argument("--d_ff", type=int, default=3072)
    model_group.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Runtime configuration
    runtime_group = arg_parser.add_argument_group("Runtime Configuration")
    runtime_group.add_argument("--batch_size", type=int, default=4)
    runtime_group.add_argument("--device", type=str, default="mps")
    runtime_group.add_argument("--dtype", type=str, default="float32")
    runtime_group.add_argument("--warm_up_steps", type=int, default=5)
    runtime_group.add_argument("--n_steps", type=int, default=10)
    
    args = arg_parser.parse_args()
    
    # Create config from args (model size preset will be applied in post_init)
    config = BenchmarkConfig.from_args(args)
    
    # Create model and input data using config
    model = get_model(config)
    input_data = generate_random_data(config)

    # Run forward and backward pass benchmarks
    print(f"\nModel configuration: {config.model_size or 'custom'}")
    print(f"Forward pass timing (mean, std):")
    print(benchmark(model, input_data, config.warm_up_steps, config.n_steps))
    
    print(f"\nBackward pass timing (mean, std):")
    print(benchmark(model, input_data, config.warm_up_steps, config.n_steps, backward=True))


if __name__ == "__main__":
    main()
