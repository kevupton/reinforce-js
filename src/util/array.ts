import * as tf from '@tensorflow/tfjs';

export function array<T = any>(length : number, value? : () => T) : T[]
export function array<T extends null | object | string | number>(length : number, value? : T) : T[] {
  const arr = Array.from(Array(length));

  if (typeof value === 'undefined') {
    return arr;
  }

  if (typeof value === 'function') {
    return arr.map(value);
  }

  return arr.map(() => value);
}

export const emptyArray = (...lengths : number[]) => array(lengths[0], () => lengths.length > 1 ? emptyArray(...lengths.slice(1)) : undefined);

export function argmax (arr : tf.Tensor) : number {
  const data = arr.dataSync();

  let index = 0;
  let value = 0;

  let i : any;
  for (i in data) {
    const v = data[i];

    if (v >= value) {
      value = v;
      index = i;
    }
  }

  return index;
}
