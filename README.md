# KAMISolver

An implementation and empirical evaluation of pathfinding algorithms such as BFS, IDDFS, ASTAR with an admissible heurstic and a not optimal heuristic. Additionally, a selection of optimizations are implemented and applied to each algorithm. The evaluation is carried out by comparing performance metrics such as optimality, optimality rate, run time and node expansions. This project is a part of my bachelor's thesis at the Albert-Ludwigs-Universität Freiburg.

The goal is to find an algorithm that is suitable for solving the game KAMI in a reasonable amount of time.


## Usage

The proper way to run the solvers is by going into the desired dataset directory with:

```bash
cd PROBLEM50-dataset/
```

or

```bash
cd PROBLEM16-dataset/
```

or 

```bash
cd PROBLEM22-dataset/
```

and then invoking the grid-kami.py script with:

```bash
python3.6 grid-kami.py [algorithm] [optimization]
```

Possible algorithm arguments:
  - bfs
  - iddfs
  - astar1
  - astar2

Possible optimization arguments:
  - normal
  - dd1
  - dd2
  - dd3
  - ssr
  - ssr-dd1
