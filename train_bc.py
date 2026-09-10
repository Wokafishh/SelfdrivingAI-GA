"""
Behaviour cloning/Imitation Learning
Inputs: Labeled data: Expert looks at state S (probably a frame of a video) and picks action A (driving forward or r/l).
Training: The model is given a state S and tries to predict the expert's action A. It is given supervised loss based on how close it is to A. criterion() + loss.backward() in pytorch.
"""
