import { Env, Opt } from './.';

export abstract class Solver<Y = number, L = number> {
  protected env: Env;
  protected opt: Opt;

  constructor(env: Env, opt: Opt) {
    this.env = env;
    this.opt = opt;
  }

  public getOpt(): any {
    return this.opt;
  }

  public getEnv(): any {
    return this.env;
  }

  /**
   * Decide an action according to current state
   * @param state current state
   * @returns decided action
   */
  public abstract decide(stateList: any): Y;
  public abstract learn(r1: L): void;
  public abstract reset(): void;
  public abstract toJSON(): object;
  public abstract fromJSON(json: {}): void;
}
