# <div align="center"> ENGR-E 599 - Monte Carlo Tree Search for robot objective planning
####  <div align="center"> ENGR-E 599 - Autonomous Robotics

<br>

###### Name: Sreesha Srinivasan Kuruvadi
###### Email: *sskuruva@iu.edu*
<br>

***
### [Monte Carlo Tree Search for robot objective planning](https://github.iu.edu/sskuruva/as_engr-599_final_project)
***

#### Problem Statement:
Objective optimization for single and mlti-robot motion.

#### Command:
<code> python main.py  </code>

#### Approach
![Monte Carlo tree search (MCTS)](https://github.iu.edu/sskuruva/as_engr-599_final_project/blob/master/media/mcts.png)


MCTS is conceptually very simple. A tree is build in an incremental and asymmetric
manner. Each iteration of MCTS consists of four steps:
- Selection: Begin with some root R, a tree policy is used to find the most urgent child of
R, then we successively select child till we reach a leaf L.
- Expansion: Unless the node L ends game, create one or more node of L and pick one of
them, call it C.
- Simulation: play random playouts from C.
- Backpropagation: Update the information of the nodes in the path from C to R using the
result of the random playouts.

#### Time Complexity
The runtime of the algorithm can be simply be computed as O(mkI/C) where m and k are the same as before, and I is the number of iterations and C is the number of cores available.
#### Memory Complexity
The memory complexity is O(mk) since in each iteration we map mk states over the cluster

#### UCT
UCT (Upper Confidence bounds applied to Trees), a popular algorithm that deals with the flaw of Monte-Carlo Tree Search, when a program may favor a losing move with only one or a few forced refutations, but due to the vast majority of other moves provides a better random playout score than other, better moves. UCT was introduced by Levente Kocsis and Csaba Szepesvári in 2006 , which accelerated the Monte-Carlo revolution in computer Go and games difficult to evaluate statically. If given infinite time and memory, UCT theoretically converges to Minimax.

#### Future considerations
##### MCTS parallelization
- Monte Carlo Trees are explored in parallel. 
- Parallel Monte Carlo Trees do not represent the same robot, but each tree represents a separate robot.

#### References
- [A Versatile Multi-Robot Monte Carlo Tree Search Planner
for On-Line Coverage Path Planning](https://arxiv.org/pdf/2002.04517.pdf)
- [Decentralised Monte Carlo Tree search for robots](https://opus.lib.uts.edu.au/bitstream/10453/97186/1/WAFR_2016_paper_50.pdf)
- Pareto Monte Carlo Tree search - Lantao lui

