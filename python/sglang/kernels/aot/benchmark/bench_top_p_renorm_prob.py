import itertools
import sgl_kernel
import torch
import triton

def torch_top_p_renorm_prob(normalized_prob, p, eps=1e-9):
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros_like(normalized_prob, dtype=torch.int32)
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    # Add eps for numerical stability
    renorm_prob_ground_truth = renorm_prob_ground_truth / (
        renorm_prob_ground_truth.sum(dim=-1, keepdim=True) + eps
    )
    return renorm_prob_ground_truth

compiled_torch_top_p_renorm_prob = torch.compile(torch_top_p_renorm_prob)

def sglang_top_p_renorm_prob(normalized_prob, p):
    return sgl_kernel.top_p_renorm_prob(normalized_prob, p)

def calculate_diff(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(
        batch_size, vocab_size, device="cuda:0", dtype=torch.float32
    )
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    renorm_prob_torch = torch_top_p_renorm_prob(normalized_prob.clone(), p)
    renorm_prob_sglang = sglang_top_p_renorm_prob(normalized_prob.clone(), p)

    try:
        torch.testing.assert_close(
            renorm_prob_torch,
            renorm_prob_sglang,
            rtol=1e-3,
            atol=1e-3,
        )
        print(
            f"✅ Implementations match for (bs={batch_size}, vocab={vocab_size}, p={p})"
        )
    except AssertionError as e:
        print(
            f"❌ Implementations differ for (bs={batch_size}, vocab={vocab_size}, p={p})"
        )
        print(e)

# Benchmark configurations
batch_size_range = [1, 8, 16, 32, 64, 99, 989]
vocab_size_range = [111, 32000, 128256]
p_range = [0.1, 0.5, 0.9]

configs = list(itertools.product(batch_size_range, vocab_size_range, p_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "p"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="top-p-renorm-prob-performance",
        args={},
    )
)
def benchmark(batch_size, vocab_size, p, provider):
    device = torch.device("cuda")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(
        batch_size, vocab_size, device=device, dtype=torch.float32
    )
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: compiled_torch_top_p_renorm_prob(normalized_prob, p)
    elif provider == "sglang":
        fn = lambda: sglang_top_p_renorm_prob(normalized_prob, p)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=32, vocab_size=32000, p=0.9)
    benchmark.run(print_data=True)