import { Mat } from 'recurrent-js';
import { SarsaExperience } from './sarsa';

export class DQNAction {

  private isComplete = new Promise(resolve => this.resolve = resolve);
  private resolve : () => void;

  constructor (
    private readonly learnFn : (experience : SarsaExperience) => void = () => {},
    private readonly rewardFn : (reward : number) => number           = reward => reward,
    readonly actionIndex : number                                     = -1,
    readonly sarsaExperience : SarsaExperience                        = {
      s0: null,
      a0: null,
      r0: null,
      s1: null,
      a1: null,
    },
  ) {}

  learn (reward : number) {
    this.sarsaExperience.r0 = this.rewardFn(reward);
    this.isComplete.then(() => {
      this.learnFn(this.sarsaExperience);
    });
  }

  next (stateVector : Mat, actionIndex : number) {
    const newAction = new DQNAction(this.learnFn, this.rewardFn, actionIndex, {
      s0: stateVector,
      a0: actionIndex,
      s1: null,
      a1: null,
      r0: null,
    });

    this.sarsaExperience.s1 = stateVector;
    this.sarsaExperience.a1 = actionIndex;

    this.resolve();

    return newAction;
  }

  reset () {
    this.sarsaExperience.s0 = null;
    this.sarsaExperience.a0 = null;
    this.sarsaExperience.r0 = null;
    this.sarsaExperience.s1 = null;
    this.sarsaExperience.a1 = null;
  }

}
