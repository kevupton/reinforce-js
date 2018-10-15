import { argmax } from '../util/array';
import { randomChoice } from '../util/random';
import { DDDQNNet } from './dddqn-net';
import { Memory } from './memory';
import * as tf from '@tensorflow/tfjs';

export class DDDQNSolver<T> {

  private readonly memory = new Memory(this.memorySize);

  private DQNNetwork = new DDDQNNet(this.stateSize, this.possibleActions.length, this.learningRate, 'DQNNetwork');
  private TargetNetwork = new DDDQNNet(this.stateSize, this.possibleActions.length, this.learningRate, 'DQNNetwork');

  constructor(
    private readonly stateSize : number[],
    private readonly learningRate : number,
    private readonly memorySize : number,
    private readonly possibleActions : T[],
  ) {
  }

  preTrain(preTrainLength : number) {
    // WE have to do the pre-training to deal with the empty memory
  }

  predict(exploreStart : number, exploreStop : number, decayRate : number, decayStep : number, state : tf.Tensor, actions : any[]) {
    const expExpTradeoff = Math.random();
    const exploreProbability = exploreStop + (exploreStart - exploreStop) * Math.exp(-decayRate * decayStep);

    let action : T;

    if (exploreProbability > expExpTradeoff) {
      action = randomChoice(this.possibleActions);
    }
    else {
      const Qs = this.DQNNetwork.predict(state.reshape([1, ...state.shape]));
      const choice = argmax(Qs);
      action = this.possibleActions[choice];
    }

    return [action, exploreProbability];
  }

  updateTargetGraph() {

    const fromVars = tf.col
  }
}
