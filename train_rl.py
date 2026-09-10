"""
Deep Reinforcement Learning / Proximal Policy Optimization
Inputs: A (randomized?) webots environment, and an algorithm for determining reward.
Training: The model is continuously given states S and has to decide action A. Actions compound in 3d space and after a certain in-world time is up the model is evaluated on how close it has made it to an arbitrary goal. It receives reward for this. Reward can be given for other arbitrary actions, such as turning right. In this way we can demotivate destructive behaviours.
We will use the WeBots + webots api for training.
"""
