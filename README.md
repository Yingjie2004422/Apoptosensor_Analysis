# Apoptosensor_Analysis

## Running BrainReg

Run BrainReg from the terminal with:

```bash
brainreg \
  /media/maryam/BELLA2024/debug/WD3_cropped.tif \
  /media/maryam/BELLA2024/debug \
  -v 1.25 1.25 1.25 \
  --orientation las \
  --atlas drosophila_wingdisc_instar3_2um \
  --brain_geometry full \
  --backend niftyreg \
  --affine-n-steps 1 \
  --affine-use-n-steps 1 \
  --freeform-n-steps 1 \
  --freeform-use-n-steps 1 \
  --bending-energy-weight 0.97 \
  --grid-spacing -1 \
  --smoothing-sigma-floating -1 \
  --smoothing-sigma-reference -1 \
  --histogram-n-bins-floating 128 \
  --histogram-n-bins-reference 128 \
  --debug \
  --pre-processing skip
```
