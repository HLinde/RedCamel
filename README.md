<!--
SPDX-FileCopyrightText: 2025 Patrizia Schoch
SPDX-FileContributor: Hannes Lindenblatt

SPDX-License-Identifier: GPL-3.0-or-later
-->

# REMI-Analysis-Validation

GUI tool to simulate Reaction Microscope detector images.

# Usage with pixi
```bash
pixi run gui
```

# Usage with mamba / conda
## Setup
- install environment with dependencies:
```bash
mamba env create
```
- activate environment:
```bash
mamba activate remigui
```

## Usage
- run GUI with:
```bash
./GUI_remi_valid.py
```
- Play around with plots and sliders!

## Updating
- pull changes:
```bash
git pull
```
- update environment:
```bash
mamba activate remigui
mamba env update
```

# Authors
- Initial implementation by Patrizia Schoch
- Maintained by Hannes Lindenblatt
