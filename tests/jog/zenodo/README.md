# Data package assembly

`assemble.py` builds the data package (13 items, NetCDF4 for gridded data, `MANIFEST.md5`) from the outputs on the data root
(`PYGLDA_V1_DATA`, default `/work/data_for_w3`) into `<data root>/zenodo_v2/`. It only reads and copies; nothing is uploaded.

    python assemble.py [--workers 28] [--only <item> ...] [--skip-global] [--manifest-only]

TWS products are packed as int16 (scale 0.1 mm, per-file offset, max packing error 0.05 mm); the conversion log and per-item sizes
are written next to the package.
