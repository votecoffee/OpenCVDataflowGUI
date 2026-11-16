# E1 validation — 2026-11-16

The directional diffuse estimator, bounded manifest import, CLI export and
diagnostic views pass **24 focused checks**. Coverage includes independently
constructed normals/albedo, exposure/power correction, isolated bright/dark
model disagreement, insufficient/degenerate light support, competing four-light
orientations, unavailable samples, tiled/cancelled execution, malformed import,
preserved outputs and identical numerical results with graphics disabled.

The `0.1.0.dev2` wheel contains five runtime modules matching their source bytes,
its MIT license and the console entry point. SHA-256:

```text
c88e029b43e94a7c6c149ec3c2aa1c6af7eeab7cdf37eb82455d10f150de9e94
```

Using the wheel directly, without installing into or changing the host
environment, all five captures were reexecuted through the CLI. All **14 raw
maps per case** exactly match the retained results, including NaN locations,
reason/support counts and per-observation evidence. The verifier permits
1e-6 absolute float tolerance for subsequent platforms; this recorded run was
exact. Source generation used Python 3.13; independent host replay used Python
3.14, OpenCV 5.0.0 and NumPy 2.5.4. See `package_verification.json`.

All five native image-review projects validate, save, reopen and load their
eight RGB views through the host's saved-project API. These are **file-only
reviews**; their PASS status confirms loading, not surface estimation. The
wheel CLI replay above checks estimation separately. No desktop interaction,
hardware acquisition or GUI training was exercised in E1.

The controls retain generic identical estimator defaults. They compare against
analytic or public CC0 material-map rendering inputs. The source material
attribution and exact transformation are retained. This does not qualify
measured surface-normal accuracy, physical height, real reflectance, motion
registration or exposure drift on acquired cameras. Acquired calibrated light
stacks and the retained host training/execution workflow remain open.

Ruff and TOML 1.0 checks pass. Package text is UTF-8 without BOM and uses LF;
external binary maps remain byte-preserved. No ML model, hardware activation,
automated CI trigger or environment installation was added.
