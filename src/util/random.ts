export function rand (a : number, b : number) {
  const min  = Math.min(a, b);
  const max  = Math.max(a, b);
  const diff = max - min;

  return Math.random() * diff + min;
}

export function randomChoice<T = any> (choices : T[]) : T {
  const index = Math.floor(choices.length * Math.random());
  return choices[index];
}
