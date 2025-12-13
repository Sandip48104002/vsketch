# vSketch: Count-Min Sketch Variants for Improved Utilization

## Overview
This repository implements and benchmarks Count-Min Sketch (CMS) data structures for network flow measurement, with a focus on reducing peak counter values and improving sketch utilization. The core idea is to distribute flow updates more evenly across the sketch, thereby minimizing hot spots (peak values) and maximizing the number of non-empty buckets.

## Key Features
- **Standard CMS**: Baseline implementation using IP addresses or flow IDs as keys.
- **Variant CMS**: Each packet is mapped to one of several possible keys per flow (e.g., using a hash variant or round-robin), distributing updates and reducing peak values.
- **Deterministic Traffic Generation**: Generates packet traces with a specified number of flows and packets, using a Zipf distribution for flow popularity. Each flow is assigned a unique, randomly chosen IPv4 address.
- **Flexible Evaluation**: Compare sketch performance using either IP addresses or flow IDs as keys, and measure the impact on utilization and peak values.
- **Notebook and Script Support**: Includes a Jupyter notebook for interactive exploration and a Python script for large-scale benchmarking.

## Motivation
In traditional Count-Min Sketches, heavy flows can create high counter values in a small number of buckets, leading to poor utilization and increased estimation error. By distributing each flow's updates across multiple possible keys (e.g., via key variants or randomized mapping), this implementation aims to:
- Lower the maximum value in any bucket (reduce peak load)
- Increase the fraction of non-empty buckets (improve utilization)
- Reduce the number of true collisions

## Usage
### Requirements
- Python 3.8+
- NumPy
- xxhash

Install dependencies:
```bash
pip install numpy xxhash
```

### Running the Benchmark
To run the main benchmark and compare standard vs. variant CMS:
```bash
python test_case.py
```
This will print utilization, collision counts, and error statistics for both keying strategies (IP and flow ID).

## File Structure
- `test_case.py` — Main benchmarking script and CMS implementations

## Customization
- Adjust the number of flows, packets, or sketch width in `test_case.py` or the notebook.
- Switch between IP and flow ID keying to observe the effect on utilization and peak values.
- Implement new sketch variants by extending the provided classes.

## References
- Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: the Count-Min Sketch and its applications. Journal of Algorithms, 55(1), 58-75.
- [Count-Min Sketch (Wikipedia)](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch)

## License
MIT License
