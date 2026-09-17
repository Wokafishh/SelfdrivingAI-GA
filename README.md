# SelfdrivingAI-GA

Python-implementation av tre olika angreppssätt för att träna en robot att navigera från punkt A till punkt B:

- **Behavioral Cloning** (`train_bc.py`) – lär sig genom att imitera demonstrationer
- **Reinforcement Learning** (`train_rl.py`) – tränas i simulatormiljön Webots genom belöningssignaler
- **Self-Supervised Learning** (`train_ssl.py`) – lär sig representationer utan labels

## Tech stack
- PyTorch
- Webots (används för RL-träningen)

### Behaviour cloning/Imitation Learning
Inputs: Labeled data: Expert looks at state S (probably a frame of a video) and picks action A (driving forward or r/l). 
Training: The model is given a state S and tries to predict the expert's action A. It is given supervised loss based on how close it is to A. criterion() + loss.backward() in pytorch.

### Deep Reinforcement Learning / Proximal Policy Optimization
Inputs: A (randomized?) webots environment, and an algorithm for determining reward.
Training: The model is continuously given states S and has to decide action Â. Actions compound in 3d space and after a certain in-world time is up the model is evaluated on how close it has made it to an arbitrary goal. It receives reward for this. Reward can be given for other arbitrary actions, such as turning right. In this way we can demotivate destructive behaviours.
We will use the WeBots + webots api for training.

### Self-Supervised / Semi-Supervised Learning
Inputs: A small batch of labeled data (state S and expert action A) and a large batch of unlabeled data (state S only).
Training: A Student model learns via gradients while a Teacher model tracks the Student as a smoothed moving average. Supervised loss compares Student predicts on clean state S to expert action A. Consistency loss compares Student predictions on noisy state S to Teacher predictions on clean state S. The Student updates through standard backpropagation, and the Teacher updates by blending its weights with the Student's new weights.
