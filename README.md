# SARSA Learning Algorithm
## AIM
Implementation of SARSA Learning Algorithm by integrating temporal differencing to find optimal policy for the given environment.

## PROBLEM STATEMENT
The given environment is the frozen lake environment where the agent must navigate from initial state to goal state avoiding holes. Various algorithms such as Value Iteration, First Visit Monte Carlo and SARSA algorithms are used to find the optimal policy for this environment. Compare these algorithms and identify the best algorithm.

## SARSA LEARNING ALGORITHM
# Step 1: 
Initial the required variables needed for the algorithm such as number of states, number of actions, lists to keep track of policies updated and the action value function.
# Step 2:
Define the select_action function which decides whether to explore or exploit and chooses an action according to the decision. 
# Step 3: 
Generate multiple learning rate and epsilon values you use for the algorithm.
# Step 4:
Iterate through episodes, compute TD target and TD error. Subsitute in the equation to find the Action Value function. Update policy choosing action with maximum value function.
# Step 5:
Return the results derived.

## SARSA LEARNING FUNCTION
### Name: BALASUBRAMANIAM L
### Register Number: 212224240020
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action=lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha,
                           min_alpha,
                           alpha_decay_ratio,
                           n_episodes)

    epsilons = decay_schedule(init_epsilon,
                              min_epsilon,
                              epsilon_decay_ratio,
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])

      while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = select_action(next_state, Q, epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]+=alphas[e]*td_error
        state, action = next_state, next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
      V=np.max(Q,axis=1)
      pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

Optimal value function
<img width="420" height="737" alt="image" src="https://github.com/user-attachments/assets/2c1d2f22-58e8-4d75-9066-d6fc385b6594" />


Policy and Success rate for the optimal policy.

<img width="617" height="102" alt="image" src="https://github.com/user-attachments/assets/e95779a0-cd9c-4231-93a0-f145e21dfc3b" />


SARSA value function
<img width="899" height="626" alt="image" src="https://github.com/user-attachments/assets/3ad0eb71-ad80-4e0e-b43f-917b27cb3cca" />


Policy and Success rate for SARSA

<img width="661" height="120" alt="image" src="https://github.com/user-attachments/assets/43e03d38-eebd-41e6-a67b-547144ee4d23" />


Comparison of the state value functions of Monte Carlo method and SARSA learning.

Monte Carlo
<img width="2500" height="777" alt="image" src="https://github.com/user-attachments/assets/47378f89-3ea4-48b5-b1c3-fa8084415941" />


SARSA 

<img width="2474" height="777" alt="image" src="https://github.com/user-attachments/assets/8429b2e4-a553-44c6-9ffe-5fa2155b294c" />


## RESULT:
Therefore, SARSA learning algorithm is implemented successfully to find optimal policy for the given environment.
