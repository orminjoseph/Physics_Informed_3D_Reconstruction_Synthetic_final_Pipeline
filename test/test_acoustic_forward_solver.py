
import torch

from dataset.acoustic_forward_solver import AcousticForwardSolver


# Use a small cube for the initial test to limit CPU memory usage.
D, H, W = 24, 24, 24

# A homogeneous velocity model in metres per second.
velocity = torch.full(
    (D, H, W),
    2000.0,
    dtype=torch.float32,
)

solver = AcousticForwardSolver(
    dt=0.0005,
    dx=10.0,
    dy=10.0,
    dz=10.0,
    n_time_steps=64,
    source_frequency=25.0,
    source_amplitude=1.0,
    device="cpu",
)

pressure, source = solver.simulate(
    velocity=velocity,
    source_location=(D // 2, H // 2, W // 2),
)

print("Pressure shape:", tuple(pressure.shape))
print("Source shape:", tuple(source.shape))
print("Pressure finite:", torch.isfinite(pressure).all().item())
print("Source finite:", torch.isfinite(source).all().item())
print("Pressure maximum:", pressure.abs().max().item())
print("Source maximum:", source.abs().max().item())
print("CFL number:", solver.last_cfl_number)

assert pressure.shape == (1, 64, D, H, W)
assert source.shape == (1, 64, D, H, W)
assert torch.isfinite(pressure).all()
assert torch.isfinite(source).all()
assert pressure.abs().max() > 0
assert source.abs().max() > 0

print("\nAcoustic forward solver smoke test passed.")
