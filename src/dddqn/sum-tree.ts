import { array } from '../util/array';

export class SumTree {
  private dataPointer = 0;
  private readonly tree : number[] = array(2 * this.capacity - 1, 0);
  private readonly data : (object | null)[] = array(this.capacity, null);

  get totalPriority () {
    return this.tree[0];
  }

  constructor (
    private readonly capacity : number,
  ) {
  }

  slice() {
    return this.tree.slice(this.tree.length - this.capacity, this.capacity);
  }

  add (priority : number, data : object) {
    const treeIndex = this.dataPointer + this.capacity - 1;

    this.data[this.dataPointer] = data;
    this.update(treeIndex, priority);
    this.dataPointer++;

    if (this.dataPointer >= this.capacity) {
      this.dataPointer = 0;
    }
  }

  update(treeIndex : number, priority : number) {
    const change = priority - this.tree[treeIndex];
    this.tree[treeIndex] = priority;

    while (treeIndex !== 0) {
      treeIndex--;
      this.tree[treeIndex] += change;
    }
  }

  getLeaf(v : number) : [ number, number, object ] {
    let parentIndex = 0;
    let leafIndex = 0;

    while (true) {
      const leftChildIndex = 2 * parentIndex + 1;
      const rightChildIndex = leftChildIndex + 1;

      if (leftChildIndex >= this.tree.length) {
        leafIndex = parentIndex;
        break;
      }
      else {
        if (v <= this.tree[leftChildIndex]) {
          parentIndex = leftChildIndex;
        }
        else {
          v -= this.tree[leftChildIndex];
          parentIndex = rightChildIndex;
        }
      }
    }

    const dataIndex = leafIndex - this.capacity + 1;
    return [ leafIndex, this.tree[leafIndex], this.data[dataIndex] ];
  }
}
