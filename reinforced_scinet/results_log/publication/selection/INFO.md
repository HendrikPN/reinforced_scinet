# Server parameters

+ 3 trainers
+ 5M episodes

# Training parameters

+ learning rate: 0.0001
+ reward clipping: 1.0e-7
+ selection discount: 0.03
+ ae discount: 10.0
+ agent discount: 1.
+ reward rescaling: 10
+  predicted actions: 1
+ training data: 200K

# Network parameters

+ Prediction model: {'env1': [64, 128, 128, 128, 128, 64, 32], 
                     'env2':  [64, 128, 128, 128, 128, 64, 32], 
                     'env3': [64, 128, 128, 128, 128, 64, 32]} 
+ Encoder model: [128, 128, 64, 32]
+ Decoder model: [32, 64, 128, 128, 128]
