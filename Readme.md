# Grammatical Path Network (GPN): You Want Cycles, Paths Is All You Need

This repository contains the code for the experiments and models presented in our paper:  
**Grammatical Path Network: You Want Cycles, Paths Is All You Need**, accepted at **LOG 2024**.  
The implementation is built with **Python, PyTorch, and CUDA**.

---

## Requirements

- **Python version**: 3.8.10  
- **PyTorch version**: 1.12.0  

Ensure that **PyTorch and CUDA** are correctly installed before running the experiments.

---

## Reproducing Experiments

### **1 TUDataset Experiment**
1. Open `TU_dataset.py`.
2. Set the dataset name in the `Name` variable (e.g., `PROTEINS`, `IMDB-B`, `NCI1`).
3. Run the experiment:
   ```bash
   python TU_dataset.py
   ```

To view the results:
```bash
python TUD_results.py
```

---

### **2 ZINC Molecular Property Prediction**
1. Open `zinc_dataset.py`.
2. Set the dataset parameters in the script.
3. Run:
   ```bash
   python zinc_dataset.py
   ```

---

### **3 Substructure Counting Experiment**
1. Open `subgraph_counting.py`.
2. Configure the parameters for counting paths and cycles.
3. Run:
   ```bash
   python subgraph_counting.py
   ```

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{piquenot2024gpn,
  title={Grammatical Path Network: You Want Cycles, Paths Is All You Need},
  author={Piquenot, Jason and Moscatelli, Aldo and Berar, Maxime and Héroux, Pierre and Raveaux, Romain and Adam, Sébastien},
  booktitle={Third Learning on Graphs Conference (LOG)},
  year={2024}
}

