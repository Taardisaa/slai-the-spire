# Overview

## Goal

### What problem is this project trying to solve?

This project aims to use reinforcement learning to train an agent to play Slay the Spire.

### What counts as success?

One practical measure of success is how far the agent can progress, such as the number of floors it can consistently clear.

### What is the scope of the implementation?

The repository covers two components: the game emulator and the RL module.

#### Game Emulator

In `src/game/action.py`:

implemented actions:
1. reward: select card reward (possibly after a combat)
2. combat: select card to play during combat
3. map: select a map node
4. campsite: upgrade a card or rest.

It doesn't implement: choosing potions; discard a card, event handling, etc.

In `src/game/const.py`, it also set some hard limits. and for me some of these limits may affect the agent's actual performance. 

1. max monsters to handle: it sets to 2.
2. card to draw per turn is set to 5.
3. some limits set to deck and cards at hand.
4. map dimensions are also set.
5. rest site health gain factory is a hardcoded one without being modified by ascension levels.

> To make it actually applicable, we'd consider putting the agent on the real game, instead of writing an emulator from scratch. Any deviation in its design would cause distribution shift.

## RL Method

### What RL algorithm is used?

The main training path uses Proximal Policy Optimization (PPO) in `src/rl/algorithms/actor_critic/master.py`. More specifically, the training update is implemented in `_update_ppo`, while the overall model design is an actor-critic one: the policy selects actions and the critic predicts state values.

Core code snippet for PPO:

```python
# Policy loss (PPO clipped)
ratio = torch.exp(log_probs_new - log_probs_old)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
loss_policy = -torch.mean(torch.min(surr1, surr2))
```

### A2C Meta-Algorithm (pseudocode)

General idea:

- The actor outputs a policy $$\pi_\theta(a \mid s)$$ and is responsible for choosing actions.
- The critic outputs a value estimate $$V_\phi(s)$$ and is responsible for estimating expected future return.
- The environment returns real rewards.
- The critic helps turn those rewards into a lower-variance learning signal for the actor.

Interaction pattern:

1. Encode the current state.
2. Actor chooses an action distribution.
3. Critic estimates the value of the current state.
4. The environment executes the action and returns reward and next state.
5. After a rollout, compute returns and advantages.
6. Update the actor with the advantage signal.
7. Update the critic to better fit the returns.

Core formulas:

- Policy: $$\pi_\theta(a_t \mid s_t)$$
- Critic: $$V_\phi(s_t)$$
- Return: $$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$$
- Advantage: $$A_t = G_t - V_\phi(s_t)$$
- Actor objective: maximize $$\log \pi_\theta(a_t \mid s_t) A_t$$
- Critic objective: minimize $$(V_\phi(s_t) - G_t)^2$$

The exact critic structure used in this repo is a value head on top of shared features:

```python
HeadValue(
    nn.Sequential(
        nn.Linear(dim_global, dim_ff_value),
        nn.ReLU(),
        nn.Linear(dim_ff_value, dim_ff_value),
        nn.ReLU(),
        nn.Linear(dim_ff_value, 1),
    )
)
```

With the current config, this is:

```python
128 -> 64 -> 64 -> 1
```

```python
initialize shared_encoder
initialize actor_head
initialize critic_head
initialize optimizer

while training:
    trajectory = []
    state = env.reset()
    done = False

    while not done:
        features = shared_encoder(state)
        action_dist = actor_head(features)
        value = critic_head(features)   # value is V(s)
        action = sample(action_dist)

        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward, value))
        state = next_state

    returns = compute_returns(trajectory)
    values = extract_values(trajectory)
    actions = extract_actions(trajectory)

    # Critic gives a baseline: an estimate of expected future return.
    advantages = returns - values

    actor_loss = policy_loss(actions, advantages)
    critic_loss = value_loss(values, returns)
    total_loss = actor_loss + critic_coef * critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Policy Model

### What is the policy architecture?

#### High-level shape

The policy architecture is hierarchical actor-critic with a shared encoder.

At a high level, the model is:

```python
state
-> shared encoder
-> primary policy head   # router
-> secondary policy head # specialist, chosen by the router if needed
-> value head            # critic
```

The shared encoder is a common deep RL design, not a repo-specific idea. It means the model first transforms the raw state into learned features, then lets both the actor and critic use those same features.

So the architecture is not:

```python
raw_state -> actor
raw_state -> critic
```

but rather:

```python
raw_state -> shared_encoder -> actor_heads
raw_state -> shared_encoder -> critic_head
```

The motivation is that even if the raw state is already numeric, it is usually not yet in the best representation for decision-making. The shared encoder is trained jointly with the actor and critic so that it learns features useful for both action selection and value prediction.

#### Shared encoder

In this repo, the shared encoder is the `Core` module. It takes the encoded game state and produces:

- transformed entity embeddings
- map representation
- a global context vector

Its structure is roughly:

```python
encoded_state
-> entity projector
-> type embeddings added to entities
-> transformer over all entities
-> split back into entity groups
-> pooled summaries + map encoding + FSM encoding
-> global projection MLP
```

More concretely:

1. Project each entity type into a shared embedding dimension.
2. Add learned type embeddings so the model can distinguish hand cards, draw pile cards, monsters, character, energy, and other sources.
3. Concatenate all entities and pass them through a transformer so they can attend to each other.
4. Encode the map separately.
5. Pool the transformed entities into summary features.
6. Concatenate pooled summaries, map features, and FSM features.
7. Project them into the final global feature vector used by the heads.

The encoder output includes both per-entity features and a global feature vector:

- `x_hand`, `x_draw`, `x_disc`, `x_deck`, `x_combat_reward`
- `x_monsters`
- `x_character`
- `x_energy`
- `x_map`
- `x_global`

The actor and critic then use different parts of this encoder output:

- the primary router head uses `x_global`
- the secondary heads use entity-level outputs such as `x_hand`, `x_monsters`, or `x_map`, together with `x_global`
- the critic uses `x_global`

#### Encoder structure

The actual shared encoder structure is closer to:

```python
Core(
    EntityProjector(dim_entity),
    nn.Embedding(num_entity_types, dim_entity),
    EntityTransformer(
        dim_embedding=dim_entity,
        dim_feed_forward=transformer_dim_ff,
        num_heads=transformer_num_heads,
        num_blocks=transformer_num_blocks,
    ),
    MapEncoder(
        kernel_size=map_encoder_kernel_size,
        embedding_dim=map_encoder_dim,
    ),
    nn.Sequential(
        nn.Linear(global_input_dim, dim_global),
        nn.ReLU(),
        nn.Linear(dim_global, dim_global),
    ),
)
```

The entity projector itself is composed of several small projection networks:

```python
EntityProjector(
    card_proj=nn.Sequential(
        nn.Linear(card_input_dim, dim_entity),
        nn.ReLU(),
        nn.Linear(dim_entity, dim_entity),
    ),
    monster_proj=nn.Sequential(
        nn.Linear(monster_input_dim, dim_entity),
        nn.ReLU(),
        nn.Linear(dim_entity, dim_entity),
    ),
    character_proj=nn.Sequential(
        nn.Linear(character_input_dim, dim_entity),
        nn.ReLU(),
        nn.Linear(dim_entity, dim_entity),
    ),
    energy_proj=nn.Sequential(
        nn.Linear(energy_input_dim, dim_entity),
        nn.ReLU(),
        nn.Linear(dim_entity, dim_entity),
    ),
    health_block_proj=nn.Sequential(
        nn.Linear(health_block_input_dim, dim_entity),
        nn.ReLU(),
        nn.Linear(dim_entity, dim_entity),
    ),
    modifier_proj=nn.Sequential(
        nn.Linear(modifier_input_dim, dim_entity),
        nn.ReLU(),
        nn.Linear(dim_entity, dim_entity),
    ),
    layer_norm=nn.LayerNorm(dim_entity),
)
```

The transformer inside `Core` is:

```python
EntityTransformer(
    nn.ModuleList([
        _EntityTransformerBlock(
            nn.MultiheadAttention(dim_entity, num_heads, batch_first=True),
            nn.LayerNorm(dim_entity),
            nn.Sequential(
                nn.Linear(dim_entity, transformer_dim_ff),
                nn.ReLU(),
                nn.Linear(transformer_dim_ff, dim_entity),
            ),
            nn.LayerNorm(dim_entity),
        )
        for _ in range(num_blocks)
    ])
)
```

The map encoder is:

```python
MapEncoder(
    nn.Conv2d(num_channels, 16, kernel_size, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 32, kernel_size, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AdaptiveAvgPool2d(1),
    nn.Linear(32, map_encoder_dim),
)
```

#### Decision flow

More concretely, the flow is:

1. The shared encoder reads the full encoded game state and produces shared features.
2. The primary policy head chooses a high-level action type among legal actions.
3. If that action type needs a concrete target or index, the model routes to a matching secondary head.
4. The secondary head chooses the actual card, monster, map node, or upgrade target.
5. The value head predicts the scalar state value.

#### Router head

The important design idea is that the first policy head acts as a router.

Its job is not to infer the broad game phase from scratch, because the state already includes that information. Instead, its job is to choose which legal action family should be taken now.

For example:

- in combat default state, the legal choices may include end turn or play a card
- at a card reward screen, the legal choices may include skip reward or select reward
- at a rest site, the legal choices may include rest or upgrade a card

So the first head answers:

> among the currently legal action types, which one should the policy take?

Then the second head answers:

> given that action type, which exact object or index should be selected?

Example flow:

1. The router chooses `CARD_PLAY`.
2. The card-selection head chooses a hand index.
3. The final action becomes: play the card at that hand position.

#### Primary actor head

This means the actor is not a single flat output layer over every possible action in the game. Instead, it is decomposed into:

- a primary router head for action categories
- several specialized secondary heads for fine-grained decisions

The heads are not fully separate end-to-end models. They share the same encoder, and only diverge at the head level. That means:

- state understanding is shared
- action selection is specialized

The primary action head is an MLP over the global state representation:

```python
HeadActionType(
    nn.Sequential(
        nn.Linear(dim_global, dim_ff_primary),
        nn.ReLU(),
        nn.Linear(dim_ff_primary, dim_ff_primary),
        nn.ReLU(),
        nn.Linear(dim_ff_primary, NUM_ACTION_CHOICES),
    )
)
```

With the current config, this is:

```python
128 -> 64 -> 64 -> 9
```

The main action categories are:

- `COMBAT_TURN_END`
- `CARD_REWARD_SKIP`
- `REST_SITE_REST`
- `CARD_PLAY`
- `CARD_DISCARD`
- `MONSTER_SELECT`
- `CARD_REWARD_SELECT`
- `CARD_UPGRADE`
- `MAP_SELECT`

#### Secondary actor heads

The secondary heads are specialized by action type:

- `HeadCardPlay`
- `HeadCardDiscard`
- `HeadCardRewardSelect`
- `HeadCardUpgrade`
- `HeadMonsterSelect`
- `HeadMapSelect`

For entity-selection actions, the secondary head scores each candidate separately using the candidate embedding plus the shared global context:

```python
HeadEntitySelection(
    nn.Sequential(
        nn.Linear(dim_entity + dim_global, dim_ff),
        nn.ReLU(),
        nn.Linear(dim_ff, dim_ff),
        nn.ReLU(),
        nn.Linear(dim_ff, 1),
    )
)
```

With the current config, the card and monster selection heads are:

```python
(64 + 128) -> 64 -> 64 -> 1
```

The map-selection head is separate because it scores map options from the map representation rather than from an entity sequence:

```python
HeadMapSelect(
    nn.Sequential(
        nn.Linear(dim_map + dim_global, dim_ff_map),
        nn.ReLU(),
        nn.Linear(dim_ff_map, dim_ff_map),
        nn.ReLU(),
        nn.Linear(dim_ff_map, MAP_WIDTH),
    )
)
```

With the current config, this is:

```python
(64 + 128) -> 64 -> 64 -> MAP_WIDTH
```

#### Critic head

The critic is another head on top of the same shared features:

```python
128 -> 64 -> 64 -> 1
```

So the full mental model is:

```python
features = shared_encoder(state)

action_type = primary_router_head(features)

if action_type needs target:
    target = corresponding_secondary_head(features)

value = critic_head(features)
```

#### Why this design

This architecture is useful because the game has several qualitatively different decision types. A flat policy would have to score all of them in one space, while this design first chooses the action family and then applies a specialist head for the fine-grained choice.

### What are the model inputs?

The policy does not consume the raw Python game objects directly. It consumes an encoded `XGameState`, which is a structured tensor bundle built in `src/rl/encoding/state.py`.

The input fields are:

```python
XGameState(
    x_hand,
    x_hand_mask_pad,
    x_draw,
    x_draw_mask_pad,
    x_disc,
    x_disc_mask_pad,
    x_deck,
    x_deck_mask_pad,
    x_combat_reward,
    x_combat_reward_mask_pad,
    x_monsters,
    x_monsters_mask_pad,
    x_monster_health_block,
    x_monster_modifiers,
    x_character,
    x_character_mask_pad,
    x_character_health_block,
    x_character_modifiers,
    x_energy,
    x_energy_mask_pad,
    x_map,
    x_fsm,
)
```

So the model input includes:

- cards in hand
- draw pile
- discard pile
- full deck
- combat reward cards
- monsters
- character state
- energy
- map state
- FSM / phase state
- padding masks for variable-length parts
- extra monster and character features such as health, block, and modifiers

At inference and training time, the policy also takes legal-action masks in addition to the encoded state:

```python
model(
    x_game_state,
    primary_mask,
    secondary_masks,
)
```

where:

- `primary_mask` says which high-level action choices are legal
- `secondary_masks` says which fine-grained targets are legal for each secondary head

So the effective policy input is not just state encoding, but state encoding plus legality masks.

### What are the model outputs?

The main batched output type is `ForwardOutput`:

```python
ForwardOutput(
    action_choices,             # (B,)
    action_choice_log_probs,    # (B,)
    secondary_indices,          # (B,)
    secondary_log_probs,        # (B,)
    values,                     # (B, 1)
)
```

This means the model outputs three kinds of information:

1. The high-level action choice.
2. The fine-grained index if that action needs a target.
3. The critic value estimate.

Conceptually, the actor output is:

```python
action = (action_type, optional_index)
```

and the critic output is:

```python
value = V(s)
```

The primary head output is an `ActionChoice`, such as:

- `COMBAT_TURN_END`
- `CARD_PLAY`
- `CARD_DISCARD`
- `MONSTER_SELECT`
- `CARD_REWARD_SELECT`
- `MAP_SELECT`

If the chosen action is terminal at the primary level, then no secondary index is needed and the output uses:

```python
secondary_index = -1
```

The final game action is assembled from these outputs:

```python
choice -> action_type
secondary_index -> index or None
Action(type=action_type, index=index)
```

The model also exposes log probabilities because PPO needs them to compute the policy ratio during updates.

### Is the action space flat or hierarchical?

It is hierarchical.

The model does not produce one single flat distribution over every possible action-instance in the game. Instead, it splits the action decision into two stages:

1. choose the action family
2. choose the concrete target/index if needed

In code, the first stage is the `ActionChoice` selection, and the second stage is routing to one of the specialized heads.

So the policy structure is:

```python
P(action | state)
= P(action_choice | state)
  * P(secondary_index | state, action_choice)
```

when a secondary decision is required.

For terminal primary actions such as `COMBAT_TURN_END`, `CARD_REWARD_SKIP`, or `REST_SITE_REST`, there is no secondary factor.

This is the main reason the implementation stores:

- primary action choice and its log probability
- secondary index and its log probability

and then combines them when needed.

Why this is hierarchical rather than flat:

- `CARD_PLAY` and `MAP_SELECT` are qualitatively different decisions
- the candidate sets are different sizes and come from different tensors
- some actions need no target at all
- masks are naturally defined per action family

So a hierarchical action space is a cleaner fit for the structure of this game than one huge flat action list.

## Reward Design

### How is reward computed?

The reward is defined in `src/rl/reward.py` and combines terminal reward with step-wise reward shaping.

If the game ends, the reward is terminal:

- loss: $$-1$$
- win: $$1 + \frac{\text{current hp}}{\text{max hp}}$$

If the game does not end, the reward is computed from state differences:

$$
r_t
= w_{hp} \cdot \Delta hp
+ w_{floor} \cdot \Delta floor
+ w_{upgrade} \cdot \Delta upgrades
+ penalty
$$

In this repo, the concrete weights are:

- $$w_{hp} = 0.025$$
- $$w_{floor} = 0.1$$
- $$w_{upgrade} = 0.5$$
- $$penalty = -0.001$$

So the reward design is shaped mainly by:

- change in character health
- change in floor progress
- change in number of upgraded cards
- a small step penalty

### What terminal rewards exist?

The terminal rewards are:

- defeat: $$-1$$
- victory: $$1 + \frac{\text{current hp}}{\text{max hp}}$$

So winning with higher remaining health gives a larger terminal reward than barely surviving.

### What shaping terms exist?

The shaping terms are based on differences between the current state and the next state.

They are:

- character health difference

$$
\Delta hp = hp_{t+1} - hp_t
$$

- floor difference

$$
\Delta floor = floor_{t+1} - floor_t
$$

- upgrade difference

$$
\Delta upgrades = upgrades_{t+1} - upgrades_t
$$

- constant step penalty

$$
-0.001
$$

So the full non-terminal shaping formula is:

$$
r_t = 0.025 \cdot \Delta hp + 0.1 \cdot \Delta floor + 0.5 \cdot \Delta upgrades - 0.001
$$

### What behaviors is the reward trying to encourage?

This reward appears to encourage four things.

1. Stay alive and preserve health.
   Because health difference is rewarded, losing less HP is better and healing is beneficial.

2. Climb floors.
   Because floor progress contributes positively, the policy is rewarded for moving forward through the run.

3. Take upgrades.
   Because newly upgraded cards give positive reward, the policy is encouraged to value upgrades as long-term improvement.

4. Avoid wasting time.
   The small per-step penalty acts like a time cost. It discourages useless or overly long trajectories and pushes the agent to make progress efficiently.

So the reward is not just asking the agent to eventually win. It also shapes the path toward winning by rewarding healthier progression, forward movement, and deck improvement, while slightly penalizing stalling.

## State Representation

### How is game state encoded?

The game state is converted into a structured tensor container called `XGameState` in `src/rl/encoding/state.py`.

This is important to separate from the shared encoder:

- `XGameState` is the structured model input
- `Core` is the neural network that transforms that input into learned features

So the pipeline is:

```python
game state
-> XGameState
-> Core
-> actor / critic heads
```

`XGameState` is not one flat vector. It is a bundle of tensors, where different parts of the game are represented separately first.

At a high level:

```python
XGameState(
    hand,
    draw_pile,
    discard_pile,
    deck,
    combat_rewards,
    monsters,
    character,
    energy,
    map,
    fsm,
    masks,
    auxiliary_features,
)
```

Then the shared encoder transforms this structured input into learned features.

An important detail is that the shared encoder does not collapse everything into a single vector only. It keeps both:

- local representations for specific entities such as cards and monsters
- global representations for whole-state decision making

So after encoding, the model has both per-entity outputs and a global summary.

### Which entities or features are represented?

The represented fields in `XGameState` are:

```python
XGameState(
    x_hand,
    x_hand_mask_pad,
    x_draw,
    x_draw_mask_pad,
    x_disc,
    x_disc_mask_pad,
    x_deck,
    x_deck_mask_pad,
    x_combat_reward,
    x_combat_reward_mask_pad,
    x_monsters,
    x_monsters_mask_pad,
    x_monster_health_block,
    x_monster_modifiers,
    x_character,
    x_character_mask_pad,
    x_character_health_block,
    x_character_modifiers,
    x_energy,
    x_energy_mask_pad,
    x_map,
    x_fsm,
)
```

In plain terms, the state representation includes:

- cards in hand
- draw pile
- discard pile
- deck
- combat reward cards
- monsters
- monster health, block, and modifiers
- character state
- character health, block, and modifiers
- energy
- map state
- FSM / current phase
- padding masks for variable-length collections

This means the state representation is not just current combat board state. It also includes hidden or longer-horizon information such as draw pile, discard pile, deck composition, map, and reward choices.

That is why the model can in principle reason about both immediate tactical decisions and longer-term planning signals.

### Is FSM or phase information included explicitly?

Yes.

The state representation includes `x_fsm`, which is an explicit encoding of the current game phase.

FSM means finite state machine. In practice, this is the current mode of the game, such as:

- combat default
- awaiting combat target
- awaiting discard target
- map
- card reward
- rest site
- game over

This means the model does not need to infer the current phase only from indirect clues. It is told explicitly which kind of decision state it is in.

This matters because the meaning of the same features changes across phases. For example:

- a hand of cards matters differently in normal combat than in a discard-selection state
- map information matters differently when choosing a node than during combat
- reward cards matter differently on a card reward screen than in a normal combat state

So yes, phase information is represented explicitly, not only implicitly.

## Action Masking And Validity

### How are invalid actions prevented?

Invalid actions are prevented by explicit boolean masks generated from the current game state in `src/rl/action_space/masks.py`.

The key idea is:

- the environment state determines what is legal
- the masks mark those legal choices
- the policy only samples from masked valid options

So legality is not left for the model to learn implicitly. Instead, the code enforces validity before sampling.

At a high level, the policy receives:

```python
model(
    x_game_state,
    primary_mask,
    secondary_masks,
)
```

Then invalid logits are replaced by $$-\infty$$ before sampling, so they receive zero probability after softmax.

So the prevention mechanism is:

1. inspect current state
2. build legality masks
3. apply masks to logits
4. sample only from valid actions

This is important because the agent is not deciding in a free-form space. It is always choosing among legal actions only.

### What masks exist?

There are two levels of masks.

#### Primary mask

The primary mask has shape:

```python
(B, NUM_ACTION_CHOICES)
```

It says which high-level `ActionChoice` values are currently legal.

Examples of primary choices include:

- `COMBAT_TURN_END`
- `CARD_PLAY`
- `CARD_DISCARD`
- `MONSTER_SELECT`
- `CARD_REWARD_SELECT`
- `MAP_SELECT`

#### Secondary masks

The secondary masks are stored as:

```python
dict[HeadType, torch.Tensor]
```

Each secondary head has its own legality mask.

The defined secondary masks are:

- `HeadType.CARD_PLAY`: valid playable cards in hand
- `HeadType.CARD_DISCARD`: valid discardable hand cards
- `HeadType.CARD_REWARD_SELECT`: valid reward cards
- `HeadType.CARD_UPGRADE`: valid upgradable cards in deck
- `HeadType.MONSTER_SELECT`: valid targetable monsters
- `HeadType.MAP_SELECT`: valid next map nodes

The output sizes are fixed by head type:

```python
CARD_PLAY          -> MAX_SIZE_HAND
CARD_DISCARD       -> MAX_SIZE_HAND
CARD_REWARD_SELECT -> MAX_SIZE_COMBAT_CARD_REWARD
CARD_UPGRADE       -> MAX_SIZE_DECK
MONSTER_SELECT     -> MAX_MONSTERS
MAP_SELECT         -> MAP_WIDTH
```

So the masking system mirrors the hierarchical action design:

- primary mask for action family
- secondary mask for fine-grained target legality

### How does masking change across different game states?

Masking changes mainly according to the FSM / current phase.

#### Combat default

Primary mask:

- `COMBAT_TURN_END = True`
- `CARD_PLAY = True` only if at least one card is affordable

Secondary mask:

- card-play mask marks only cards whose cost is less than or equal to current energy

#### Combat awaiting card target

Primary mask:

- `MONSTER_SELECT = True`

Secondary mask:

- monster mask marks all available monsters

#### Combat awaiting discard target

Primary mask:

- `CARD_DISCARD = True`

Secondary mask:

- discard mask marks all cards currently in hand

#### Card reward screen

Primary mask:

- `CARD_REWARD_SKIP = True`
- `CARD_REWARD_SELECT = True` only if reward cards exist and the deck is not full

Secondary mask:

- reward mask marks the offered reward cards

#### Rest site

Primary mask:

- `REST_SITE_REST = True`
- `CARD_UPGRADE = True` only if at least one card in the deck is not upgraded

Secondary mask:

- upgrade mask marks only cards whose names do not end with `+`

#### Map

Primary mask:

- `MAP_SELECT = True`

Secondary mask:

- map mask marks only reachable next nodes
- on the first floor, it marks valid starting nodes
- later, it marks nodes reachable from the current node

So the masks are phase-sensitive and state-sensitive. They do not just depend on the broad game mode; they also depend on details such as energy, deck fullness, card upgrade status, available monsters, and reachable map branches.

## Training Loop

### How are rollouts collected?

The main training loop is in `src/rl/algorithms/actor_critic/master.py`.

Rollouts are collected in parallel using multiple worker processes. Each worker runs its own environment, and the master process batches model inference across all currently active environments.

The flow is roughly:

```python
start worker processes

while training:
    reset all workers

    while some workers are still active:
        collect current states from active workers
        tensorize states into XGameState batch
        build primary and secondary masks
        run model forward pass
        send chosen actions back to workers
        receive next states and rewards
        store transitions
```

Each stored `Transition` contains:

- tensorized state
- masks
- chosen primary action
- chosen secondary index
- action log probabilities
- value estimate
- reward

An important implementation detail is that the code stores pre-tensorized states and masks, not just raw game objects. That makes later PPO recomputation simpler and more efficient.

### How are updates computed?

After trajectories are collected, the trainer builds a `TrajectoryBatch`.

The update pipeline is:

1. collect trajectories from workers
2. compute returns and advantages using GAE
3. normalize advantages
4. split the batch into minibatches
5. recompute current log probabilities and values
6. apply the PPO clipped objective
7. apply value loss and entropy regularization
8. backpropagate and step the optimizer

The GAE part is conceptually:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

and then:

$$
R_t = A_t + V(s_t)
$$

The PPO policy update is:

$$
r_t(\theta) = \exp\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{old}(a_t \mid s_t)\right)
$$

$$
L_{policy} = -\mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]
$$

The value loss is either clipped or unclipped depending on config. The entropy term is added to encourage exploration.

So the total update is actor-critic with PPO-style policy stabilization.

### What are the important training hyperparameters?

The main training hyperparameters from `src/rl/algorithms/actor_critic/config.yml` are:

```python
num_episodes = 1_000_000
log_every = 10
save_every = 100

gamma = 0.99
lam = 0.95

coef_value = 0.5
coef_entropy_max = 0.015
coef_entropy_min = 0.010
coef_entropy_elbow = 25_000

max_grad_norm = 0.5
num_envs = 24

clip_eps = 0.20
clip_value_loss = True
minibatch_size = 512
num_epochs = 4

optimizer = Adam(lr=1e-4)
```

The most important ones conceptually are:

- `gamma`: reward discounting
- `lam`: GAE smoothing
- `clip_eps`: PPO clipping strength
- `coef_value`: weight of critic loss
- `coef_entropy_*`: exploration pressure over training
- `num_envs`: rollout parallelism
- `minibatch_size` and `num_epochs`: how much PPO reuses collected data
- `max_grad_norm`: gradient clipping for stability

### What gets logged or checkpointed?

Training logs are written with TensorBoard `SummaryWriter` under:

```python
experiments/<exp_name>/
```

The trainer logs:

- policy loss
- value loss
- entropy loss
- entropy coefficient
- total trajectory reward
- average trajectory length
- greedy evaluation reward
- greedy evaluation length

The trainer also periodically saves:

```python
experiments/<exp_name>/model.pth
```

and copies the training config into the experiment directory.

So the experiment directory acts as the main artifact store for:

- checkpoints
- config snapshot
- TensorBoard logs

## Evaluation

### How is the trained agent evaluated?

There are two clear evaluation paths in the repo.

The first is an in-training greedy evaluation episode in `src/rl/algorithms/actor_critic/master.py`. During training, the master process periodically runs one evaluation episode with `sample=False` and logs its reward and length.

The second is the standalone evaluation script `src/rl/test_agent.py`, which:

- loads a saved model from `experiments/<exp_name>/`
- runs one or more games
- can render the full game step by step
- can run quietly and print summary statistics
- can optionally show grouped card-play probabilities during combat

So the repo supports both lightweight online evaluation during training and post-training manual inspection.

### Are evaluations greedy, sampled, scripted, or benchmark-based?

The main built-in evaluation is greedy.

- `master.py` uses `sample=False` for evaluation episodes
- `test_agent.py` also supports greedy mode via `--greedy`

The test script can also run with sampling, so it can be used in either deterministic or stochastic mode.

I do not currently see a formal benchmark suite for the active PPO path in the repo. There is no dedicated evaluation harness with fixed seeded scenarios, regression-style benchmark tables, or standard leaderboard-style metrics.

There is an older scripted evaluation module under `src/rl/legacy/evaluation.py`, but that appears tied to the legacy path rather than the current PPO pipeline.

### What metrics matter most?

The metrics that seem to matter most in this repo are:

- final floor reached
- final remaining health
- total evaluation reward
- episode length

During training, the code explicitly logs:

- policy loss
- value loss
- entropy loss
- total trajectory reward
- average trajectory length
- greedy evaluation reward
- greedy evaluation length

For higher-level project success, floor reached is probably the clearest metric, because it best matches overall run progress. Final health is also useful because the reward function explicitly values health preservation.

## Results

### What results are available now?

The repo contains the machinery to produce results, but I do not currently see committed experiment artifacts or a written summary of achieved performance.

What is available in code:

- TensorBoard logging under `experiments/<exp_name>/`
- saved model checkpoints
- greedy evaluation during training
- post-training rollout testing via `src/rl/test_agent.py`

What I do not currently see in the repository itself:

- published benchmark numbers
- comparison across seeds
- comparison between PPO and other methods
- a stable claimed win rate or average floor

### What evidence shows progress?

The strongest evidence of progress built into the codebase would come from:

- increasing evaluation reward over training
- longer average trajectories that correspond to deeper runs
- higher final floor in post-training test runs
- healthier final states in those runs

So the codebase is set up to generate evidence of progress, but the evidence itself appears to live in runtime experiment outputs rather than in checked-in documentation.

### What is still unknown or not yet measured?

Several important things are still unclear from the repository alone:

- how strong the trained PPO agent currently is
- whether training is stable across seeds
- whether the hierarchical policy clearly outperforms simpler alternatives
- whether the reward shaping helps or hurts final gameplay quality
- whether hard masking materially improves final performance, not just training stability

So the implementation is fairly concrete, but the empirical picture is still incomplete without actual experiment outputs.

## Project Maturity

### Which parts are active and current?

The active and current path appears to be the hierarchical actor-critic PPO stack under:

- `src/rl/algorithms/actor_critic/`
- `src/rl/models/actor_critic.py`
- `src/rl/action_space/`
- `src/rl/encoding/`

This path has the clearest design investment:

- shared encoder
- hierarchical policy heads
- masking system
- parallel rollout workers
- PPO update logic
- test-time agent runner

### Which parts are legacy or experimental?

The legacy path is under `src/rl/legacy/` and `src/rl/legacy/dqn_algorithm/`.

That code appears to represent an older DQN-based approach, along with older evaluation helpers. It is still useful for historical context, but it does not look like the main direction of the repo now.

The simpler single-threaded actor-critic training script in `src/rl/algorithms/actor_critic/train.py` looks more like a debugging or learning-oriented implementation than the main production training path.

### What are the biggest current limitations?

The main limitations I would flag are:

- no checked-in results summary
- limited visible automated evaluation for the active PPO path
- environment scope is narrower than the full real game
- some hard-coded environment limits may constrain what the policy can learn
- performance quality depends heavily on hand-designed masks and reward shaping

So the code architecture is reasonably mature in design, but the project still feels research/prototype-like in evaluation and scope.

## Environment Scope

### Which parts of the game are implemented?

From the action space and FSM handling, the implemented decision types include:

- combat card selection
- combat target selection
- combat turn end
- card reward selection or skip
- map node selection
- rest or card upgrade at rest sites

So the environment is not combat-only. It also includes map progression, post-combat rewards, and campfire choices.

### What has been simplified?

Several things appear simplified relative to the full game.

- The action space is restricted to a small set of supported decisions.
- Rest site actions are only rest or upgrade.
- The code uses hard limits such as max hand size, max deck size, max reward size, map width, and max monsters.
- Some game systems from the real game do not appear in the current action interface, such as broader event handling or potion use.

So the repo should be understood as a simplified Slay the Spire-like environment, not a faithful reproduction of the full game.

### What assumptions or constraints shape learning?

The main assumptions and constraints are:

- the agent acts only inside the implemented action space
- invalid actions are removed by hard masks
- the reward is shaped rather than purely outcome-based
- the environment has hard-coded structural limits
- the learned policy is optimized for this emulator, not the original game directly

This last point matters: if the emulator differs meaningfully from the real game, then a policy trained here may face distribution shift when transferred outside this environment.

## Open Questions

### What is still unclear after reading the code?

The main unclear points for me are:

- how well the current PPO setup performs in practice
- whether the reward shaping coefficients are well tuned
- whether the hierarchical policy is empirically better than a flatter alternative
- how much the hard masks help final gameplay quality versus only helping optimization
- how much the environment simplifications limit generalization

### What should be verified experimentally?

The most useful experiments to run next would be:

- compare greedy vs sampled evaluation performance
- compare PPO against the simpler actor-critic path
- ablate reward shaping terms
- ablate hard masking or vary masking strategy where feasible
- run multiple seeds and compare stability
- track average floor reached and final health over time

### What would be the best next files or modules to inspect?

The most useful next modules to inspect are:

- `src/rl/reward.py` for reward shaping
- `src/rl/action_space/masks.py` for legality design
- `src/rl/models/core.py` for state-feature construction
- `src/rl/models/actor_critic.py` for head routing and final outputs
- `src/rl/algorithms/actor_critic/master.py` for the full PPO loop
- `src/rl/test_agent.py` for post-training behavior inspection
