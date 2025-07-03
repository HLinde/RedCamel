<!--
SPDX-FileCopyrightText: 2025 Patrizia Schoch
SPDX-FileContributor: Hannes Lindenblatt

SPDX-License-Identifier: GPL-3.0-or-later
-->
# Remi Detector Calculation for Monitoring Electrons

GUI tool to simulate Reaction Microscope detector images.
# Usage with pixi
```bash
pixi run redcamel
```
pixi can be found here: https://pixi.sh/latest/#installation

# Usage with uv
```bash
uv run redcamel
```
uv can be found here: https://docs.astral.sh/uv/getting-started/installation/

# Example Outputs
![Electron Wiggles](Electrons.png)
![Ion fragmentation](Ions.png)

# Usage with mamba / conda
## Setup
- install environment with dependencies:
```bash
mamba env create
```
## Usage
- activate environment:
```bash
mamba activate redcamel
```
- run GUI with:
```bash
python src/redcamel/remi_gui.py
```
- Play around with plots and sliders!

## Updating
- pull changes:
```bash
git pull
```
- update environment:
```bash
mamba activate redcamel
mamba env update
```

# Authors
- Initial implementation by Patrizia Schoch
- Maintained by Hannes Lindenblatt
