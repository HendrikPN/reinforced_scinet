# Server parameters

+ 21 workers, 2 predictors, 1 trainer each
+ 3M episodes

# Training parameters

+ glow: 0.1
+ gamma: 0.01
+ softmax: 0.5
+ learning rate: 0.00005
+ reward clipping: 1.0e-7

# Network parameters

+ DPS model: {'env1': [128, 128, 128, 128, 64, 32], 
              'env2':  [128, 128, 128, 128, 64, 32], 
              'env3': [128, 128, 128, 128, 64, 32]}
