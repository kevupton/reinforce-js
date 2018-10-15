import { Graph, Mat, Net, NetOpts, Utils } from 'recurrent-js';
import { Env } from '../env';
import { Solver } from '../solver';
import { DQNOpt } from './dqn-opt';
import { SarsaExperience } from './sarsa';

export class C51Agent extends Solver {

  // Env
  public numberOfStates: number;
  public numberOfActions: number;

  // Opts
  public numberOfHiddenUnits: Array<number>;

  public readonly epsilonMax: number;
  public readonly epsilonMin: number;
  public readonly epsilonDecayPeriod: number;
  public readonly epsilon: number;

  public readonly gamma: number;
  public readonly alpha: number;
  public readonly doLossClipping: boolean;
  public readonly lossClamp: number;
  public readonly doRewardClipping: any;
  public readonly rewardClamp: any;
  public readonly experienceSize: number;
  public readonly keepExperienceInterval: number;
  public readonly replaySteps: number;

  // Local
  protected net: Net;
  protected previousGraph: Graph;
  protected shortTermMemory: SarsaExperience = { s0: null, a0: null, r0: null, s1: null, a1: null };
  protected longTermMemory: Array<SarsaExperience>;
  protected isInTrainingMode: boolean;
  protected learnTick: number;
  protected memoryIndexTick: number;

  batch_size : number;

  target_model : any;
  model : any;

  constructor (env: Env, opt: DQNOpt) {
    super(env, opt);

    this.numberOfHiddenUnits = opt.get('numberOfHiddenUnits');

    this.epsilonMax = opt.get('epsilonMax');
    this.epsilonMin = opt.get('epsilonMin');
    this.epsilonDecayPeriod = opt.get('epsilonDecayPeriod');
    this.epsilon = opt.get('epsilon');

    this.experienceSize = opt.get('experienceSize');
    this.gamma = opt.get('gamma');
    this.alpha = opt.get('alpha');
    this.doLossClipping = opt.get('doLossClipping');
    this.lossClamp = opt.get('lossClamp');
    this.doRewardClipping = opt.get('doRewardClipping');
    this.rewardClamp = opt.get('rewardClamp');

    this.keepExperienceInterval = opt.get('keepExperienceInterval');
    this.replaySteps = opt.get('replaySteps');

    this.isInTrainingMode = opt.get('trainingMode');

    self.batch_size = 32
    self.observe = 2000
    self.explore = 50000
    self.frame_per_action = 4
    self.update_target_freq = 3000
    self.timestep_per_train = 100 # Number of timesteps between training interval


    self.num_atoms = num_atoms # 51 for C51
      self.v_max = 30 # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
    self.v_min = -10 # -0.1*26 - 1 = -3.6
    self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
    self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    self.model = None
    self.target_model = None

    self.memory = deque()
    self.max_memory = 50000 # number of previous transitions to remember

    this.reset();
  }

  public reset(): void {
    this.numberOfHiddenUnits = this.opt.get('numberOfHiddenUnits');
    this.numberOfStates = this.env.get('numberOfStates');
    this.numberOfActions = this.env.get('numberOfActions');

    const netOpts: NetOpts = {
      architecture: {
        inputSize: this.numberOfStates,
        hiddenUnits: this.numberOfHiddenUnits,
        outputSize: this.numberOfActions
      }
    };
    this.net = new Net(netOpts);

    this.learnTick = 0;
    this.memoryIndexTick = 0;

    this.shortTermMemory.s0 = null;
    this.shortTermMemory.a0 = null;
    this.shortTermMemory.r0 = null;
    this.shortTermMemory.s1 = null;
    this.shortTermMemory.a1 = null;

    this.longTermMemory = [];
  }

  update_target_model() {
    this.target_model.set_weights(this.model.get_weights());
  }

  get_action () {
    if (Math.random() <= this.epsilon) {

    }
  }

  /**
   * Decide an action according to current state
   * @param state current state
   * @returns index of argmax action
   */
  public decide(state: Array<number>): number {
    const stateVector = new Mat(this.numberOfStates, 1);
    stateVector.setFrom(state);

    const actionIndex = this.epsilonGreedyActionPolicy(stateVector);

    this.shiftStateMemory(stateVector, actionIndex);

    return actionIndex;
  }

  protected epsilonGreedyActionPolicy(stateVector: Mat): number {
    if (Math.random() < this.currentEpsilon()) { // greedy Policy Filter
      return Utils.randi(0, this.numberOfActions);
    } else {
      // Q function
      const actionVector = this.forwardQ(stateVector);
      return Utils.argmax(actionVector.w); // returns index of argmax action
    }
  }

}

def get_action(self, state):
"""
Get action from model using epsilon-greedy policy
"""
if np.random.rand() <= self.epsilon:
#print("----------Random Action----------")
action_idx = random.randrange(self.action_size)
else:
action_idx = self.get_optimal_action(state)

return action_idx

def get_optimal_action(self, state):
"""Get optimal action for a state
"""
z = self.model.predict(state) # Return a list [1x51, 1x51, 1x51]

z_concat = np.vstack(z)
q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)

// Pick action with the biggest Q value
action_idx = np.argmax(q)

return action_idx

def shape_reward(self, r_t, misc, prev_misc, t):

# Check any kill count
if (misc[0] > prev_misc[0]):
r_t = r_t + 1

if (misc[1] < prev_misc[1]): # Use ammo
r_t = r_t - 0.1

if (misc[2] < prev_misc[2]): # Loss HEALTH
r_t = r_t - 0.1

return r_t

# save sample <s,a,r,s'> to the replay memory
def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
if self.epsilon > self.final_epsilon and t > self.observe:
self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

if len(self.memory) > self.max_memory:
self.memory.popleft()

# Update the target model to be same with model
  if t % self.update_target_freq == 0:
self.update_target_model()

# pick samples randomly from replay memory (with batch_size)
  def train_replay(self):

num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
replay_samples = random.sample(self.memory, num_samples)

state_inputs = np.zeros(((num_samples,) + self.state_size))
next_states = np.zeros(((num_samples,) + self.state_size))
m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
action, reward, done = [], [], []

for i in range(num_samples):
state_inputs[i,:,:,:] = replay_samples[i][0]
action.append(replay_samples[i][1])
reward.append(replay_samples[i][2])
next_states[i,:,:,:] = replay_samples[i][3]
done.append(replay_samples[i][4])

z = self.model.predict(next_states) # Return a list [32x51, 32x51, 32x51]
z_ = self.target_model.predict(next_states) # Return a list [32x51, 32x51, 32x51]

# Get Optimal Actions for the next states (from distribution z)
optimal_action_idxs = []
z_concat = np.vstack(z)
q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
q = q.reshape((num_samples, action_size), order='F')
optimal_action_idxs = np.argmax(q, axis=1)

# Project Next State Value Distribution (of optimal action) to Current State
for i in range(num_samples):
if done[i]: # Terminal State
# Distribution collapses to a single point
Tz = min(self.v_max, max(self.v_min, reward[i]))
bj = (Tz - self.v_min) / self.delta_z
m_l, m_u = math.floor(bj), math.ceil(bj)
m_prob[action[i]][i][int(m_l)] += (m_u - bj)
m_prob[action[i]][i][int(m_u)] += (bj - m_l)
else:
for j in range(self.num_atoms):
Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
bj = (Tz - self.v_min) / self.delta_z
m_l, m_u = math.floor(bj), math.ceil(bj)
m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

loss = self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, nb_epoch=1, verbose=0)

return loss.history['loss']

# load the saved model
def load_model(self, name):
self.model.load_weights(name)

# save the model which is under training
def save_model(self, name):
self.model.save_weights(name)

if __name__ == "__main__":

# Avoid Tensorflow eats up GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

game = DoomGame()
game.load_config("../../scenarios/defend_the_center.cfg")
game.set_sound_enabled(True)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.init()

game.new_episode()
game_state = game.get_state()
misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
prev_misc = misc

action_size = game.get_available_buttons_size()

img_rows , img_cols = 64, 64
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

# C51
num_atoms = 51

state_size = (img_rows, img_cols, img_channels)
agent = C51Agent(state_size, action_size, num_atoms)

agent.model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)
agent.target_model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)

x_t = game_state.screen_buffer # 480 x 640
x_t = preprocessImg(x_t, size=(img_rows, img_cols))
s_t = np.stack(([x_t]*4), axis=2)    # It becomes 64x64x4
s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

is_terminated = game.is_episode_finished()

# Start training
epsilon = agent.initial_epsilon
GAME = 0
t = 0
max_life = 0 # Maximum episode life (Proxy for agent performance)
life = 0

# Buffer to compute rolling statistics
life_buffer, ammo_buffer, kills_buffer = [], [], []

while not game.is_episode_finished():

loss = 0
r_t = 0
a_t = np.zeros([action_size])

# Epsilon Greedy
action_idx  = agent.get_action(s_t)
a_t[action_idx] = 1

a_t = a_t.astype(int)
game.set_action(a_t.tolist())
skiprate = agent.frame_per_action
game.advance_action(skiprate)

game_state = game.get_state()  # Observe again after we take the action
is_terminated = game.is_episode_finished()

r_t = game.get_last_reward()  #each frame we get reward of 0.1, so 4 frames will be 0.4

if (is_terminated):
if (life > max_life):
max_life = life
GAME += 1
life_buffer.append(life)
ammo_buffer.append(misc[1])
kills_buffer.append(misc[0])
print ("Episode Finish ", misc)
game.new_episode()
game_state = game.get_state()
misc = game_state.game_variables
x_t1 = game_state.screen_buffer

x_t1 = game_state.screen_buffer
misc = game_state.game_variables

x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

r_t = agent.shape_reward(r_t, misc, prev_misc, t)

if (is_terminated):
life = 0
else:
life += 1

#update the cache
prev_misc = misc

# save the sample <s, a, r, s'> to the replay memory and decrease epsilon
agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

# Do the training
if t > agent.observe and t % agent.timestep_per_train == 0:
loss = agent.train_replay()

s_t = s_t1
t += 1

# save progress every 10000 iterations
if t % 10000 == 0:
print("Now we save model")
agent.model.save_weights("models/c51_ddqn.h5", overwrite=True)

# print info
state = ""
if t <= agent.observe:
state = "observe"
elif t > agent.observe and t <= agent.observe + agent.explore:
state = "explore"
else:
state = "train"

if (is_terminated):
print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ LIFE", max_life, "/ LOSS", loss)

# Save Agent's Performance Statistics
if GAME % agent.stats_window_size == 0 and t > agent.observe:
print("Update Rolling Statistics")
agent.mavg_score.append(np.mean(np.array(life_buffer)))
agent.var_score.append(np.var(np.array(life_buffer)))
agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

# Reset rolling stats buffer
life_buffer, ammo_buffer, kills_buffer = [], [], []

# Write Rolling Statistics to file
with open("statistics/c51_ddqn_stats.txt", "w") as stats_file:
stats_file.write('Game: ' + str(GAME) + '\n')
stats_file.write('Max Score: ' + str(max_life) + '\n')
stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
stats_file.write('var_score: ' + str(agent.var_score) + '\n')
stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')
