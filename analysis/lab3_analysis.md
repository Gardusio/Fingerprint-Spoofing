# Applying PCA
What do you observe? What are the effects on the class distributions? Can you spot the different clusters inside each class?

# Applying LDA
What do you observe? Do the classes overlap? Compared to the histograms of the 6 features you computed in Laboratory 2, is LDA finding a good direction with little class overlap?

# Classify with LDA and projected mean diff threshold
Now try changing the value of the threshold. What do you observe? Can you find values that improve the classification accuracy?

# Classify with LDA after PCA
Analyze the performance as a function of the number of PCA dimensions m. What do you observe? Can you find values of m that improve the accuracy on the validation set? Is PCA beneficial for the task when combined with the LDA classifier?