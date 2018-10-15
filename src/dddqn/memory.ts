import { emptyArray } from '../util/array';
import { rand } from '../util/random';
import { SumTree } from './sum-tree';

export class Memory {
  private PER_e = 0.01;
  private PER_a = 0.6;
  private PER_b = 0.4;
  private PER_b_increment_per_sampling = 0.001;
  private absolute_error_upper = 1;
  private tree : SumTree;

  constructor(
    capacity : number
  ) {
    this.tree = new SumTree(capacity);
  }

  store(experience : object) {
    let max_priority = Math.max(...this.tree.slice());

    if (max_priority === 0) {
      max_priority = this.absolute_error_upper;
    }

    this.tree.add(max_priority, experience);
  }

  sample(n : number) {
    const memory_b : object[][] = [];
    const b_idx : (number | undefined)[] = emptyArray(n);
    const b_ISWeights = emptyArray(n, 1);
    const prioritySegment = this.tree.totalPriority / n;

    this.PER_b = Math.min(1, this.PER_b + this.PER_b_increment_per_sampling);

    const pMin = Math.min(...this.tree.slice()) / this.tree.totalPriority;
    const maxWeight = (pMin * n) ** (-this.PER_b);

    for (let i = 0; i < n; i++) {
      const a = prioritySegment * i;
      const b = prioritySegment * (i + 1);
      const value = rand(a, b);

      const [ index, priority, data ] = this.tree.getLeaf(value);
      const samplingProbabilities = priority / this.tree.totalPriority;

      b_ISWeights[i][0] = Math.pow(n * samplingProbabilities, -this.PER_b) / maxWeight;
      b_idx[i] = index;

      const experience = [data];
      memory_b.push(experience);
    }

    return [b_idx, memory_b, b_ISWeights];
  }

  batchUpdate(treeIdx : number[], absErrors : number[]) {
    absErrors = absErrors.map(error => error + this.PER_e);
    const clippedErrors = absErrors.map(error => Math.min(error, this.absolute_error_upper));
    const ps = clippedErrors.map(error => Math.pow(error, this.PER_a));

    for (let i = 0; i < treeIdx.length; i++) {
      const ti = treeIdx[i];
      const p = ps[i];

      this.tree.update(ti, p);
    }
  }
}
