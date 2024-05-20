# Evolutionary Algorithm for the Traveling Salesman Problem

This repository contains an implementation of an evolutionary algorithm designed to solve the Traveling Salesman Problem (TSP). The algorithm incorporates fitness sharing, 2-opt and 3-opt local search methods, and adaptive schemes to improve performance and convergence. This project was developed as part of the Genetic Algorithms and Evolutionary Computing course at KU Leuven University. Special thanks to the course instructors and fellow students for their valuable feedback and contributions.

## Features

1. **Fitness Sharing**: Implements a fitness sharing scheme during the elimination phase to maintain diversity in the population.
2. **Local Search Methods**: Utilizes 2-opt and 3-opt local search algorithms to enhance solution quality.
3. **Adaptivity and Self-adaptivity**: Features adaptive mechanisms for parameters like mutation rates and local search intensity to dynamically improve algorithm performance.

## File Structure

- `Code.py`: Contains the implementation of the evolutionary algorithm.
- `Explanation and results.pdf`: Provides a detailed explanation of the algorithm, its design, parameter choices, and performance results on various TSP instances.
