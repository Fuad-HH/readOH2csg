# Benchmarks

```bash
cmake -S . -B build/bench -DOMEGAH2CSG_ENABLE_BENCH=ON && cmake --build build/bench -j
python bench/run_bench.py --build build/bench --out bench/_results/run.json \
    [--baseline bench/_results/main.json] [--max-size 1000000]
```

- `make_mesh.cpp` writes jittered box meshes (1e4 to 1e7 triangles), cached in
  `bench/_meshes/`. They have no wall tags, so they only exercise the OpenMC
  path. Real tagged meshes for the DEGAS2 path are listed in `meshes.toml` and
  kept outside git.
- `copy_counter.cpp` is a Kokkos Tools plugin that counts copies between
  memory spaces (`cross_space_copies` in the JSON). On CUDA, this includes
  the copies omega-h makes while reading the mesh.
- Results report per-stage median wall time over 5 runs (after one warm-up),
  peak RSS and the copy count. With `--baseline`, any stage more than 15% slower
  fails. The t(1e5)/t(1e4) scaling ratio is reported only, not gated.
