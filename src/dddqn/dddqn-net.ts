import * as tf from '@tensorflow/tfjs';

export class DDDQNNet {

  private readonly inputs = tf.input({
    shape: [null, ...this.state_size],
    dtype: 'float32',
    name: 'inputs',
  });

  private readonly ISWeights = tf.input({
    shape: [null, 1],
    dtype: 'float32',
    name: 'IS_weights',
  });

  private readonly actions = tf.input({
    shape: [null, this.action_size],
    dtype: 'float32',
    name: 'actions',
  });

  private readonly target_Q = tf.input({
    shape: [null],
    dtype: 'float32',
    name: 'target',
  });

  private readonly conv1 = tf.layers.conv2d({
    filters: 32,
    kernelSize: [8, 8],
    strides: [4, 4],
    padding: 'valid',
    kernelInitializer: tf.initializers.glorotNormal({}),
    name: 'conv1',
  });

  private readonly conv2 = tf.layers.conv2d({
    filters: 64,
    kernelSize: [4, 4],
    strides: [2, 2],
    padding: 'valid',
    kernelInitializer: tf.initializers.glorotNormal({}),
    name: 'conv2',
  });

  private readonly conv3 = tf.layers.conv2d({
    filters: 128,
    kernelSize: [4, 4],
    strides: [2, 2],
    padding: 'valid',
    kernelInitializer: tf.initializers.glorotNormal({}),
    name: 'conv1',
  });

  private conv1_out = tf.layers.elu({
    name: 'conv1_out',
  }).apply(this.conv1.apply(this.inputs));

  private conv2_out = tf.layers.elu({
    name: 'conv2_out',
  }).apply(this.conv2.apply(this.conv1_out));

  private conv3_out = tf.layers.elu({
    name: 'conv3_out',
  }).apply(this.conv3.apply(this.conv2_out));

  private flatten = tf.layers.flatten().apply(this.conv3_out);

  private value_fc = <tf.Tensor>tf.layers.dense({
    units: 512,
    activation: 'elu',
    kernelInitializer: tf.initializers.glorotNormal({}),
    name: 'value_fc',
  }).apply(this.flatten);

  private value = tf.layers.dense({
    units: 1,
    activation: null,
    kernelInitializer: tf.initializers.glorotNormal({}),
    name: 'value',
  }).apply(this.value_fc);

  private advantage_fc = tf.layers.dense({
    units: 512,
    activation: 'elu',
    kernelInitializer: tf.initializers.glorotNormal({}),
    name: 'advantage_fc',
  }).apply(this.flatten);

  private advantage = tf.layers.dense({
    units: this.action_size,
    activation: null,
    kernelInitializer: tf.initializers.glorotNormal({}),
  }).apply(this.advantage_fc);


  private readonly subtract = tf.sub(this.advantage, tf.mean(this.advantage, 1, true));
  readonly output = tf.add(this.value, this.subtract);
  readonly Q = tf.sum(tf.mul(this.output, this.actions), 1);
  readonly absolute_errors = tf.abs(tf.sub(this.target_Q, this.Q));
  readonly loss = tf.mean(tf.mul(this.ISWeights, tf.squaredDifference(this.target_Q, this.Q)));
  readonly optimizer = new tf.RMSPropOptimizer(this.learning_rate).minimize(() => this.loss);


  private readonly model = tf.model({
    inputs: this.inputs,
    outputs: this.output,
  });

  constructor (
    readonly state_size : number[],
    readonly action_size : number,
    readonly learning_rate : number,
    readonly name : string,
  ) {
  }

  predict(input : any) {
    return <tf.Tensor>this.model.predict(input);
  }

  save() {
    this.model.save('./brain2.json');
  }
}
