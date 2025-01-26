import pytest
import torch
from sgl_kernel import sampling_scaling_penalties

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 65]
vocab_sizes = [2048, 4096, 8192, 16384, 32768, 32767]
dtypes = [torch.float32, torch.half, torch.bfloat16]


@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("vocab_size", vocab_sizes)
@pytest.mark.parametrize("dtype", dtypes)
def test_sampling_scaling_penalties(batch_size, vocab_size, dtype):
    device = torch.device("cuda")
    rtol = 1e-3
    atol = 1e-3

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    scaling_penalties = (
        torch.rand(batch_size, vocab_size, device=device, dtype=dtype) + 0.5
    )

    ref_output = torch.where(
        logits > 0, logits / scaling_penalties, logits * scaling_penalties
    )

    kernel_output = sampling_scaling_penalties(logits, scaling_penalties)

    torch.testing.assert_close(
        kernel_output,
        ref_output,
        rtol=rtol,
        atol=atol,
        msg=f"Failed for batch_size={batch_size}, vocab_size={vocab_size}, dtype={dtype}",
    )


if __name__ == "__main__":
    pytest.main([__file__])
