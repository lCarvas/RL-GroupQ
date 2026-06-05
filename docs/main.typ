#import "@preview/ilm:2.1.0": *
#import "@preview/zero:0.6.1": *

#set text(lang: "en")
#set-round(mode: "figures", precision: 4, pad: false, direction: "nearest", ties: "away-from-zero")
#set-num(exponent: (sci: 3), omit-unity-mantissa: true)
#show ref: it => {
  let el = it.element
  if el == none or el.func() != heading { it } else {
    let l = it.target
    link(l, [#el.body (p. #counter(page).at(el.location()).first())])
  }
}

#show: ilm.with(
  title: [Reinforcement Learning Group Q],
  authors: (
    "Diogo Carvalho - 20221935",
    "Ricardo Pereira - 20250343",
    "Yehor Malakhov - 20221691",
  ),
  date: datetime.today(),
  logo: image("/assets/IMS_Logo.png"),
)

#counter(page).update(1)

= Introduction

== Sepsis

Sepsis is a life-threatening clinical condition with extensive physiological and biochemical abnormalities. The Third
International Consensus currently defines sepsis as "organ dysfunction caused by a dysregulated host response to
infection".#super[@sepsis-pathophysiology] Approximately 49 million people are affected by sepsis every year, and it is
estimated that 11 million deaths are caused by the disease, accounting for up to 19.7% of all deaths worldwide.

Clinical understanding is continuously developing toward an immunological perspective, but it has not yet been possible
to transform knowledge into evidence-based practice for the effective treatment of sepsis.

The use of machine learning methods to evaluate clinical data may help to identify further markers and patient
subclasses which are associated with severity and outcome.

== Reinforcement Learning for Sepsis Treatment

Reinforcement Learning (RL) is applied to sepsis as a sequential decision making problem, where an agent decides optimal
treatment policies by observing patient states and selecting interventions such as IV fluid and vasopressor dosages to
maximise long-term patient outcomes.

RL models can use historical ICU (Intensive Care Unit) data to learn personalised treatment strategies, with patient
conditions represented as states and treatment decisions represented as actions.#super[@choudhary2024icusepsis,
  @rl-sepsis-treatment] Subsequent studies have refined these methods and reported promising results.

However, most existing approaches rely on discretized dosages because dosage distributions are highly uneven, which
limits clinical usefulness, providing broad treatment ranges rather than precise dosages.

= Algorithms

== Metrics

We measure the performance of our algorithms by evaluating the average return and survival rate, the latter of which
corresponds to the percentage of episodes where the return is greater than 0.

== Hyperparameter Tuning

To achieve the best results possible, we performed hyperparameter tuning for all our algorithms using Optuna, an
open-source library for hyperparameter optimisation.#super[@optuna] We defined a hyperparameter grid for each algorithm
and ran multiple trials to find the best hyperparameters that maximise the average return. The details of the
hyperparameter tuning process and the best hyperparameters found for each algorithm are described in the respective
sections below.

== Configuration A <config-a>

Configuration A consists of a discrete environment with 716 states, corresponding to the clinical state of the patient,
and 25 actions corresponding to 5 vasopressor levels and 5 IV-fluid dose levels. An episode ends when states 713 or 714
are reached. These states correspond to the death or survival of a patient, and give a reward of 0 or 1, respectively.
The discount factor for this environment is 1, thus, we won't be mentioning it in the hyperparameter tuning sections, as
it is not a hyperparameter that we can tune.

The environment is discrete and has a small state space, along with this, we were provided with the transition
probabilities and reward function of the environment, which makes it possible to use model-based algorithms, in our
case, policy iteration. Policy iteration can rapidly compute the optimal policy for this environment, given the
transition probabilities and reward function, this provided us with a benchmark to compare our other configuration A
algorithm against. Along with this, we implemented a model-free algorithm. When faced with the choice between Q-Learning
and SARSA, we chose Q-Learning, as it is an off-policy algorithm. Off-policy algorithms learn the value of the optimal
policy independently of the agent's actions, which makes more sense in our environment, as it would be like learning the
best practice from different doctors' policies, instead of learning the best practice from a single doctor. In theory,
given infinite exploration, both Q-Learning and SARSA would converge to the optimal policy. However, in practice, due to
the limited number of episodes we can run, and the fact that we cannot guarantee infinite exploration, it is more likely
that Q-Learning will converge to a better policy than SARSA, hence our choice.

The reason behind the choice of these algorithms is that since the environment is discrete and has a small state space,
we can compute the optimal policy through tabular methods, along with this, we were provided with the transition
probabilities and reward function, which makes it possible to use model-based algorithms.

Besides that, we tried to understand the environment by visualising the results of the random policy on episodes. In
@rand-respond, you can see how most of the states are as likely to be visited, and that the returns observed are quite
low, which was expected.

=== Policy Iteration

Given a Markov Decision Process (MDP) defined by the tuple $(S, A, P, R, gamma)$, where $S$ is the set of states, $A$ is
the set of actions, $P$ is the transition probability function, $R$ is the reward function, and $gamma$ is the discount
factor, we want to find an optimal policy $pi$ that maximises the return. We can do this through policy iteration, which
consists of two steps: policy evaluation and policy improvement. In policy evaluation, we compute the value function
$V_pi$ for a given policy $pi$. The value function gives us an estimate of how good it is to be in each state when
following the policy. In policy improvement, we update the policy to be greedy with respect to the value function, that
is, we choose the action that maximises the return based on the current value function. We repeat these two steps until
convergence, that is, until the policy does not change anymore.

We can compute the value function for a given policy by solving the Bellman Expectation Equation for $V_pi$, which is as
follows:

$
  V_pi (s) = sum_a pi (s, a) sum_(s prime) p (s prime | s, a) [R(s, a, s prime) + gamma V_pi (s prime)]
$
$
  V_pi (s) = sum_a pi (s, a) sum_(s prime) P(s, a, s prime) R(s, a) + gamma sum_a pi (s, a) sum_(s prime) P(s, a, s prime) V(s prime)
$
$
  V_pi (s) = sum_a pi (s, a) R(s, a) + gamma sum_(s prime) sum_a pi (s, a) P(s, a, s prime) V (s prime)
$
$
  V_pi (s) = r_pi (s) + gamma sum_(s prime) P_pi (s, s prime) V(s prime)
$

Where $s in S$ and $a in A$.

We can now solve this by iterating over the set of states and updating the value function until convergence. However, we
can also solve this as a linear system of equations.

For all states at once:

$
  V_pi = r_pi + gamma P_pi V_pi
$
$
  V_pi - gamma P_pi V_pi = r_pi
$
$
  (I - gamma P_pi) V_pi = r_pi
$
$
  A V_pi = r_pi
$

Then we could solve this as a $a x = b$ linear system, but iff $A$ is invertible.

==== Non Invertible Matrices

A matrix M is non-invertible, also known as singular, if its kernel is non-trivial, that is:

$
  exists v != 0 : M v = 0
$

Row stochastic matrices, such as $P_pi$, have the property that the sum of each row is 1, that is:

$
  sum_(s prime) P_pi (s, s prime) = 1, forall s in S
$
$
  P_pi 1 = 1
$ <ppi-sum-property>

With this property in mind (@ppi-sum-property), and taking into account the definition of A, we can prove that, for
$gamma = 1$, $mat(1, 1, ..., 1, delim: "[")^T$ is part of the kernel of A:

$
  A 1 = (I - P_pi) 1 = I 1 - P_pi 1 = 1 - 1 = 0
$

Therefore, no matter the policy, A will always be singular when $gamma = 1$.

This is a problem, since for our environment, $gamma = 1$, and thus we cannot compute $V_pi$ by solving the linear
system of equations. However, we can use least squares to compute an approximate solution to the system, which is what
we do in our implementation.

The average return obtained with this implementation, when tested on #num(50000, math: false, exponent: auto) different
seeds, was #num(
  0.7868,
  math: false,
  exponent: auto,
), with a survival rate of 82.5% (@poit-results).

As it is an approximation, this implementation will not reach the optimal policy. However, it converges extremely fast,
that is, in 4 iterations. The results in @poit-results show two clear groups of well-treated and poorly-treated
patients. Note the preference of the policy for shorter episodes.

=== Q-Learning

Q-Learning is an off-policy, model-free algorithm that optimises the action-value function $Q(s, a)$, which represents
the expected return of taking action $a$ in state $s$.#super[@qlearning-paper] The update rule for Q-Learning is as
follows:

$
  Q(s, a) = Q(s, a) + alpha [r + gamma max_(a prime) Q(s prime, a prime) - Q(s, a)]
$

==== Q-Learning Hyperparameter Tuning <qlearning-hyperparameter-tuning>

In order to tune the hyperparameters of our Q-Learning implementation, we used Optuna to perform a search over the
following hyperparameter grid:

#figure(
  table(
    columns: 3,
    table.header[*Hyperparameter*][*Value/Range*][*Domain*],
    [Learning Rate], [$10^(-5), 0.01$], [Log],
    [Starting Exploration Rate], [1.0], [],
    [Minimum Exploration Rate], [$0.01, 0.1$], [Log],
    [Exploration Fraction], [$0, 1$], [Linear],
  ),
)

Each Optuna trial ran the Q-Learning algorithm for #num(400000, math: false, exponent: auto) episodes, and evaluated the
average return on #num(15000, math: false, exponent: auto) different seeds. The best hyperparameters found were the
following:

#figure(
  table(
    columns: 2,
    table.header[*Hyperparameter*][*Value*],
    [Learning Rate], [$#num(0.18484003928169826)$],
    [Starting Exploration Rate], [1.0],
    [Minimum Exploration Rate], [$#num(0.00012850334707145372)$],
    [Exploration Fraction], [$#num(0.6712650723031417)$],
  ),
)

With these hyperparameters, we obtained an average return of #num(0.7074, math: false, exponent: auto), with a survival
rate of 81% (@qlearning-performance). The difference between the average return and the survival rate can be attributed
to some episodes having a high length. Besides that, both the mean return value and the episode length converged after
#num(200000, math: false, exponent: auto) episodes.

=== Configuration A Results

As can be seen in @config-a-algorithm-comparison, both algorithms achieved better mean return than the random policy,
with Policy Iteration performing better than Q-Learning.

== Configuration B

Configuration B consists of a 47-dimensional continuous environment. These 47 dimensions correspond to the 47 clinical
variables used in "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive
care" article by Komorowski et al.#super[@Komorowski2018] Along with this, certain random events were added to the
environment. The events are the following:

- Occasional Monitor Malfunction, which corrupts the current episode's with Gaussian noise.
- Occasional Missing Lab Values, which makes a subset of the 47 clinical variables unavailable for the remainder of the
  episode.
- Rare Sudden Patient Death, which causes the patient to die, regardless of the actions taken by the agent.

These events were added to make the environment more realistic, modelling real-life situations such as equipment
calibration issues, lab analysis delays, and unexpected patient deterioration, as a side effect, these events also make
the environment more challenging for the agent. Just like in @config-a, the discount factor is 1.

As this is a continuous environment, with a large state space, the algorithms we implemented in configuration A are not
suitable for this environment. As such, we had to implement another two algorithms for this configuration. We
implemented one value-based algorithm, Deep Q-Network (DQN), and one policy-based algorithm, Proximal Policy
Optimisation (PPO). Q-Learning gave us good results in configuration A, so we decided to implement DQN, which is a
Q-Learning extension for continuous environments. We also wanted to implement a policy-based algorithm, and PPO is a
stable algorithm due to its clipping mechanism, which prevents large updates to the policy, and thus prevents the
algorithm from diverging. This is particularly important in our environment, as it contains the aforementioned random
events, which can make the learning process more unstable.

For this configuration, we implemented the following algorithms:

- Deep Q-Network (DQN)
- Proximal Policy Optimisation (PPO)

The algorithms were implemented using the Stable Baselines3 library. You can find the documentation for the DQN and PPO
on the Stable Baselines3 Documentation website. #super[@stable-baselines3, @stable-baselines3-docs]

=== Deep Q-Network (DQN)

The Deep Q-Network (DQN) algorithm#super[@dqn-paper], introduced by Mnih et al. (2015), extends classical Q-learning to
high-dimensional, continuous state spaces by approximating the action-value function $Q(s,a)$ with a deep neural
network, referred to as the Q-network. In the context of Configuration B, the agent receives as input a 47-dimensional
state vector comprising the clinical variables defined by Komorowski et al. (2018), and must learn a policy that maps
these observations to a discrete action space representing treatment decisions.

==== Value Function and Training Objective
The Q-network is parameterised by weights $theta$ and is trained to minimise the mean squared Bellman error:

$
  L(theta) = E_((s,a,r,s prime)) ~ D [(r + gamma max_(a prime) Q(s prime,a prime;theta^-) - Q(s,a;theta))^2]
$

where $gamma$ is the discount factor, and $theta$ are the weights of a periodically-updated target network. The use of a
separate target network stabilises training by decoupling the targets from the rapidly updating online network,
preventing the feedback loop that arises when the same network generates both predictions and targets.

==== DQN Hyperparameter Tuning <dqn-hyperparameter-tuning>

The hyperparameter grid used for tuning the DQN algorithm is the following:

#figure(
  table(
    columns: 3,
    table.header[*Hyperparameter*][*Value/Range*][*Domain*],
    [Learning Rate], [$#num(1e-5), #num(1e-2)$], [Log],
    [Buffer Size], [$10^4, 5 times 10^5$], [Linear],
    [Learning Starts], [$10^3, 10^4$], [Linear],
    [Batch Size], [$32, 64, 96, 128, 160, 192, 224, 256$], [Categorical],
    [Train Frequency], [$1, 4, 8, 16$], [Categorical],
    [Polyak Update Coefficient], [$#num(1e-3), #num(1.0)$], [Log],
    [Target Update Interval], [$#num(1), 10^3$], [Linear],
    [Exploration Fraction], [$#num(0.01), #num(0.5)$], [Linear],
    [Starting Exploration Rate], [1.0], [],
    [Minimum Exploration Rate], [$#num(0.01), #num(0.1)$], [Linear],
  ),
)

All the unmentioned hyperparameters were left with their default values, as defined in the Stable Baselines3
documentation.

Unlike in @qlearning-hyperparameter-tuning, Random Search was used instead of TPESampler. Each Optuna trial ran the DQN
algorithm for #num(2000000, math: false, exponent: auto) timesteps, which is roughly equivalent to #num(
  200000,
  math: false,
) episodes, and evaluated the average return every #num(500000, math: false, exponent: auto) timesteps on #num(
  1000,
  math: false,
) different seeds. The best hyperparameters found were the following:

#figure(
  table(
    columns: 2,
    table.header[*Hyperparameter*][*Value*],
    [Learning Rate], [$#num(7.238086291181395e-05)$],
    [Buffer Size], [$#num(295826, exponent: auto)$],
    [Learning Starts], [$1274$],
    [Batch Size], [$32$],
    [Train Frequency], [$1$],
    [Polyak Update Coefficient], [$#num(0.20416470207182763)$],
    [Target Update Interval], [$216$],
    [Exploration Fraction], [$#num(0.31521633315131015)$],
    [Starting Exploration Rate], [1.0],
    [Minimum Exploration Rate], [$#num(0.01768127184943912)$],
  ),
)

After tuning the hyperparameters, we trained another DQN algorithm with the best found hyperparameters for #num(
  2000000,
  math: false,
  exponent: auto,
) timesteps, which ended up being #num(243777, math: false, exponent: auto) episodes, and evaluated the average return
on #num(15000, math: false, exponent: auto) different seeds. This was done for two reasons: first, due to a coding
mistake, the survival rate was not being returned, and thus, we only had the average return; this same mistake was also
made during the hyperparameter tuning phase for Proximal Policy Optimisation. Second, we wanted to evaluate on more
seeds to get a better estimate of the average return and survival rate.

With these hyperparameters, we obtained an average return of #num(0.6565, math: false, exponent: auto), with a survival
rate of 70.1% (@dqn-performance). Interestingly, DQN was not improving for the first #num(
  50000,
  math: false,
  exponent: auto,
) episodes, yet right after it showed an unexpected jump in mean return. Notably, the algorithm did not fully converge
on the #num(243777, math: false, exponent: auto)th episode, but we could not explore it further, because of
computational limitations.

=== Proximal Policy Optimisation (PPO)

Proximal Policy Optimisation#super[@ppo-paper] is an on-policy, model-free policy gradient algorithm that improves
training stability by constraining how much the policy is permitted to change in a single update step. At each
iteration, PPO collects a batch of experience under the current policy and then performs multiple gradient updates on
that data but rather than maximising the raw policy gradient objective, which can cause destructively large parameter
updates, it optimises a clipped surrogate objective that acts as a conservative lower bound on the expected return of
the updated policy. The surrogate objective is defined as:
$
  L^"CLIP" (theta) = E_t [min (r_t (theta) A_t, "clip" (r_t (theta), 1 - epsilon, 1 + epsilon) A_t)]
$

where

$
  r_t(theta) = pi_theta(a_t | s_t) / pi_(theta_"old")(a_t | s_t)
$

is the probability ratio between the new and old policy, $A_t$ is the estimated advantage at timestep $t$, and $epsilon$
is the clipping hyperparameter. The min operator ensures that when the advantage is positive, meaning the action was
better than expected, the policy update is capped at a ratio of $1 - epsilon$, preventing the algorithm from
over-exploiting a single favourable observation. Symmetrically, when the advantage is negative, the ratio is floored at
$1 + epsilon$. This clipping mechanism removes the incentive to move $r_t$ outside the interval
$[1 - epsilon, 1 + epsilon]$, effectively keeping the updated policy within a trust region of the old policy and
preventing the large, destabilising parameter updates that naive policy gradient methods are susceptible to.

==== PPO Hyperparameter Tuning

The hyperparameter grid used for tuning the PPO algorithm is the following:

#figure(
  table(
    columns: 3,
    table.header[*Hyperparameter*][*Value/Range*][*Domain*],
    [Learning Rate], [$#num(1e-5), #num(1e-2)$], [Log],
    [Number of Steps], [$128, 256, 512, 1024, 2048$], [Categorical],
    [Batch Size], [$4, 8, 16, 32, 64$], [Categorical],
    [Number of Epochs], [$3, 10$], [Linear],
    [GAE Lambda], [$0.85, 0.99$], [Linear],
    [Clipping Range], [$0.1, 0.3$], [Linear],
    [Entropy Coefficient], [$#num(1e-4), 0.1$], [Log],
    [Value Function Coefficient], [$0.1, 1$], [Linear],
    [Max Gradient Norm], [$0.3, 4.0$], [Linear],
  ),
)

All the unmentioned hyperparameters were left with their default values, as defined in the Stable Baselines3
documentation.

Just like in @dqn-hyperparameter-tuning, Random Search was used instead of TPESampler. Each Optuna trial ran the PPO
algorithm for #num(2000000, math: false, exponent: auto) timesteps, which is roughly equivalent to #num(
  200000,
  math: false,
) episodes, and evaluated the average return every #num(500000, math: false, exponent: auto) timesteps on #num(
  1000,
  math: false,
) different seeds. The best hyperparameters found were the following:

#figure(
  table(
    columns: 2,
    table.header[*Hyperparameter*][*Value*],
    [Learning Rate], [#num(0.00010989552401964294)],
    [Number of Steps], [#num(512)],
    [Batch Size], [#num(16)],
    [Number of Epochs], [#num(3)],
    [GAE Lambda], [#num(0.9066120957607506)],
    [Clipping Range], [#num(0.24846562726824667)],
    [Entropy Coefficient], [#num(0.0046146978454535996)],
    [Value Function Coefficient], [#num(0.3588452603186762)],
    [Max Gradient Norm], [#num(1.4238598958760802)],
  ),
)

After tuning the hyperparameters, we trained another PPO algorithm with the best found hyperparameters for #num(
  2000000,
  math: false,
  exponent: auto,
) timesteps, which ended up being #num(234773, math: false, exponent: auto) episodes, and evaluated the average return
on #num(
  15000,
  math: false,
  exponent: auto,
) different seeds. The reasons for doing this were mentioned in the @dqn-hyperparameter-tuning section.

With these hyperparameters, we obtained an average return of #num(0.6688, math: false, exponent: auto), with a survival
rate of 68.5% (@ppo-performance). Compared to DQN, PPO demonstrated faster learning, converging approximately after
#num(70000, math: false, exponent: auto) episodes. Notably, this was close to the number of episodes required for DQN to
begin learning. In other words, even though PPO was faster to learn, DQN understood the problem better.

=== Configuration B Results

As can be seen in @config-b-algorithm-comparison, both algorithms achieved better mean return than the random policy,

= Conclusions

== Final Results

#figure(
  table(
    columns: 5,
    table.cell(
      rowspan: 2,
    )[],
    [*Policy Iteration*],
    [*Q-Learning*],
    [*DQN*],
    [*PPO*],
    table.cell(
      colspan: 2,
    )[Configuration A],
    table.cell(
      colspan: 2,
    )[Configuration B],
    [Mean Return],
    [0.7868],
    [0.7074],
    [0.6565],
    [0.6688],
    [STD Return Survivors],
    [0.0474],
    [0.1083],
    [0.0485],
    [0.0172],
    [STD Return Non-Survivors],
    [0.0434],
    [0.0930],
    [0.0447],
    [0.0173],
    [Survival Rate],
    [82.5%],
    [81%],
    [70.1%],
    [68.5%],
  ),
)

The results between the two configurations are not directly comparable, as they are different environments; nonetheless,
the results of configuration B being worse than the results of configuration A can be attributed to the fact that
configuration B is a much more complex environment, with a much continuous state space instead of a discrete one, and
thus it is expected that the algorithms would perform worse on it.

== Limitations and Future Work

We believe that conducting only 30 hyperparameter search trials for DQN and PPO, that is, 30 unique hyperparameter
combinations, was likely insufficient to identify the optimal hyperparameters for these algorithms. This limitation is
particularly relevant given that the trials were conducted using Random Search rather than TPESampler. The decision to
restrict the number of trials to 30 was driven by computational constraints, as the tuning process for these algorithms
required a total of approximately 28 hours to complete. This is something that we could look to improve in future work,
as it is likely that better hyperparameters could be found with a more extensive search, and maybe a better defined
hyperparameter grid.

#show bibliography: set text(0.85em)
#show bibliography: set heading(numbering: "1.")
#show bibliography: set par(leading: 0.65em, justify: false, linebreaks: auto)
#bibliography("refs.bib")

= Annexes

#figure(
  image("/assets/config_a_random.png"),
  caption: [Environment Respond to Random Policy],
) <rand-respond>
#figure(
  image("/assets/policy_iteration.png"),
  caption: [Policy Iteration Results],
) <poit-results>
#figure(
  image("/assets/qlearning.png"),
  caption: [Q-Learning Performance Analysis],
) <qlearning-performance>
#figure(
  image("/assets/config_a_algorithm_comparison.png"),
  caption: [Configuration A Algorithm Comparison],
) <config-a-algorithm-comparison>
#figure(
  image("/assets/dqn.png"),
  caption: [DQN Performance Analysis],
) <dqn-performance>
#figure(
  image("/assets/ppo.png"),
  caption: [PPO Performance Analysis],
) <ppo-performance>
#figure(
  image("/assets/config_b_comparison.png"),
  caption: [Configuration B Algorithm Comparison],
) <config-b-algorithm-comparison>
