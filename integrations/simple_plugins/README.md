# Optional Simple Plugin source

The included [bin-picking package](bin_picking/PACKAGE_README.md) supplies the
existing trained suction tool and the explicitly loaded simulation grid example.
Install it into the app environment with compatible dependencies already present:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation .\integrations\simple_plugins\bin_picking
```

The public example runs from this included source. Source hashes
are in `bin_picking_sha256.json`. Original code is MIT. ROS packages remain
separately built under `integrations/ros2`; their binary runtime is not installed
into the desktop app. Reinstall/restart the app after updating plugin source.
