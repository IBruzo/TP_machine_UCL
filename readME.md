# README: Model Scripts and Optimization

## Overview
This document outlines the scripts used for running and optimizing models across three parts of the project, along with the utility of specific scripts for graphing.

---

## **Scripts for Running Models**

### **Part 1**
- **Script:** `brainy.py`
- **Purpose:** Executes the model for Part 1.

### **Part 2**
- **Script:** `brainnyBruzo.py`
- **Purpose:** Executes the model for Part 2.

### **Part 3**
- **Script:** `brainy.py`
- **Purpose:** Executes the model for Part 3.

---

## **Scripts for Optimizing Parameters**

### **Part 1**
- **Script:** `n_features_model.py`
- **Purpose:** Optimizes parameters for the **correlation filter**.

### **Part 2**
- **Script:** `prueba.py`
- **Purpose:** Optimizes parameters for the **MLP model**.

### **Part 3**
- **Script:** `prueba.py`
- **Purpose:** Optimizes parameters for the **MLP model**, incorporating the **CNN**.

---

## **Graphing Script**

### **Part 3**
- **Script:** `hipo.py`
- **Purpose:** Implements useful graphs to visualize results.

---

## **Usage**
1. Run the model scripts in sequence for the respective parts of the project.
2. Use the parameter optimization scripts before running the model scripts to ensure optimal performance.
3. For Part 3, leverage the `hipo.py` script to generate visual insights into the results.

---

## **Directory Structure**
```
project/
│
├── part1/
│   ├── brainy.py
│   ├── n_features_model.py
│
├── part2/
│   ├── brainnyBruzo.py
│   ├── prueba.py
│
├── part3/
│   ├── brainy.py
│   ├── prueba.py
│   ├── hipo.py
│
└── README.md
```

---

## **Contact**
For questions or issues, please reach out to the development team.