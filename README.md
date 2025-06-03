# Yakemon - Pokemon Battle Agent Project ⚡

## Introduction to Project
This was a project to build an AI that can effectively beat humans at Pokemon battles made to contend in YAICON, a competition from YAI.
Our goal was simple but extremely challenging: achieve 60%+ win rate against members of Yonsei's Pokemon club 'Posenyeon' battle division. Pokemon battles might look simple, but they're incredibly complex - type matchups, status effects, stat changes, switching timing, probability calculations... there's a lot going on under the hood.

## Technical Approach

### Reinforcement Learning using DDDQN and Custom Environment
- **Main Algorithm**: DDDQN (Dueling Double Deep Q-Network)
- **State Vector**: 1153 dimensions encoding every aspect of battle
- **Action Space**: 6 actions (4 moves + 2 switches)
- **Reward Function**: Complex strategic reward system
- **Custom Environment**: We didn't use the standard Poke-env, and used a self-made environment that nearly used all of the logics in a Pokemon battle

### Massive State Space of 1153 Dimensions

We went all-out on state representation to make sure our AI knows everything that's happening:

```python
# Battle global state: 51 dimensions
#   - Turn count: 1
#   - Weather effects (4 types × 6 one-hot): 24  
#   - Field effects (4 types × 5 one-hot): 20
#   - Room effects (6 one-hot): 6

# Side field state (my + enemy): 52 dimensions  
#   - Hazards: stealth rock, spikes, toxic spikes
#   - Screens: reflect, light screen, aurora veil

# Pokemon state (6 pokemon × 177 each): 1062 dimensions
#   - Species, ability, moves, PP
#   - Types, HP, stat boosts
#   - Status conditions, volatile effects
#   - Position, charging states, etc.

# Active Pokemon move types: 72 dimensions
# Total: 1153 dimensions
```

Every single piece of information that could affect battle decisions gets encoded into numbers. We utilized one-hot encoding most of the time.
As Pokemon has a enormous state space considering all species of possible characters, we had to create a large state space for better representation.
Since we used a custom environment, we had a hard time coding this without error... The simulator correcting took most of our project time.

### The Reward Function Hell

This was our biggest challenge. Started simple with sparse rewards such as "win = +5, lose = -5" but the AI just spammed healing moves and ignored type matchups.
As a minor difference in rewards resulted in extremely variant results, we had to tweak things very slowly. As a result we found out with sparse rewards the early learning process was deteriorated alot; thus for a better start regarding learning the game we had to supply a better reward function.

This resulted in us tweaking rewards in these kinds of directions :
- **Type effectiveness**: Massive bonuses for super effective hits, penalties for resisted attacks
- **Strategic switching**: Rewards for good type matchups, penalties for switching into weaknesses  
- **Status condition usage**: Penalties for using status moves on already-statused targets
- **Ability awareness**: Penalties for using moves that get nullified by abilities
- **Stat boosting timing**: Rewards for setup when faster than opponent
- **Damage dealing**: Scaled rewards based on damage percentage
All of these rewards weren't implemented in the final project, but we tried using all of these criteria throughout the reward shaping process.


```python
# Example reward calculations
if was_effective == 2:  # 4x damage
    reward += 2.0
elif was_effective == -2:  # 1/4 damage  
    reward -= 2.0

if switched_into_immunity:
    reward += 1.5
elif switched_into_4x_weakness:
    reward -= 1.0
```

## Results and Analysis, Discoveries

Unfortunately as it took us a long time adjusting and correcting the simulator to represent a real Pokemon battle, we didn't exactly have enough time to adjust the rewards or learning process for the optimal agent.
If the simulator was running quickly we would have tried to implement Inverse RL or Monte Carlo Tree Search to get the optimal battle agent, but we didn't have enough time.
Thus, with just DDDQN and training in a 1e4 scale of episodes, we achieved 50% over the random base AI.

Analysis on results:
- **Simulator coding took too much time**: We didn't have enough time to implement other strategies
- **Action space encoding problems**: We implemented a non-constant action space, which probably didn't lead to optimal results

The AI is learning some cool behaviors:
- **Type matchup optimization**: Immediately switches out of bad matchups
- **Setup timing**: Uses stat boosts when faster than opponent  
- **Ability awareness**: Avoids moves that get nullified
- **Hazard play**: Uses entry hazards strategically

## Running the Code

### Setup
```bash
git clone https://github.com/your-repo/yakemon.git
cd yakemon
pip install -r requirements.txt
```

### Training
```bash
# Train DDDQN agent
python train_dddqn.py --episodes 10000 --batch-size 128

# Evaluate against base AI
python evaluate.py --model ./models/best_model.pth --games 100
```

### Key Files
```
yakemon/
├── env/battle_env.py              # Gym environment
├── RL/
│   ├── get_state_vector.py        # 1153-dim state creation
│   ├── reward_calculator.py       # Complex reward function  
│   └── agent_choose_action.py     # RL agent actions
├── agent/dddqn_agent.py           # DDDQN implementation
└── context/                       # Battle state management
```

## Team

- **김동욱**: Project lead, simulator integration
- **김재후**: RL algorithms, DDDQN implementation
- **박민우**: RL algorithms, simulator integration
- **김현중**: Reward function design
