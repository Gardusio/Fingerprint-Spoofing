### Analyze the first two features. 
*What do you observe? Do the classes overlap? If so, where? Do the classes show similar mean for the first two features? Are the variances similar for the two classes? How many modes are evident from the histograms (i.e., how many “peaks” can be observed)?*

#### Feature 1
- The histogram for feature 1 shows both classes have a good approximation of a gaussian as the distribution for this feature. There's one mode around the mean for both. 

- There's a large overlap around zero, that is approximately the mean of both classes [G_mean: 0.0005, C_mean: 0.003]. Genuines values for f1 can take all the values of Counterfeits and more. Genuine set shows it can reach greather absolute values for this feature (Fake's never reach around 4/-4) <br>

- The Genuine set shows more variance for this feature [G_var: 1,43, C_var: 0.57].

Taken alone, this feature can't provide a good decision boundary since there's a large overlap between the two classes.

#### Feature 2
- The histogram for feature 2 shows both classes have a good approximation of a gaussian as the distribution for this feature. There's one mode around the mean for both, but the situation is dual with respect of feature 1

- There's a large overlap around zero, that is approximately the mean of both classes [G_mean: -0.0085, C_mean: 0.018]. Counterfeits values for f2 can take all the values of Genuines and more. Counterfeits set shows it can reach greather absolute values for this feature (Fake's never reach around 4/-4)<br>

- The Counterfeits set shows more variance for this feature [G: 0.58, C: 1.42].

Taken alone, this feature can't provide a good decision boundary.


### Analyze the third and fourth features. 
- What do you observe? Do the classes overlap? If so, where? Do the classes show similar mean for these two features? Are the variances similar for the two classes? How many modes are evident from the histograms?

#### Feature 3
- The histogram for feature 3 shows both classes have a good approximation of (the same, shifted) gaussian as the distribution for this feature. There's one mode around the mean for both, but this time the mean are significatively different.

- There's some overlap around 0, but overall the two distributions are well separated. Genuines spans in [-2, 3] while Counterfeits in [-3, 2)

- Genuines mean for feature 3 is 0,66, Counterfeits mean is -0, 68 (same shape)

- Genuine variance for feature 3 is 0.54, 0,55 (same shape)

#### Feature 4

- The histogram for feature 4 shows both classes have a good approximation of (the same, shifted) gaussian as the distribution for this feature. There's one mode around the mean for both, but this time the mean are significatively different. 

- The situation is dual with respect of feature 3

- There's some overlap around 0, but overall the two distributions are well separated. Genuines spans in [-3, 2) while Counterfeits in (-2, 3]

- Genuines mean for feature 3 is -0,66, Counterfeits mean is 0,67 (same shape)

- Genuine variance for feature 3 is 0.55, 0,53 (same shape)

### Analyze the last two features. 
- What do you observe? Do the classes overlap? If so, where? How many modes are evident from the histograms? How many clusters can you notice from the scatter plots for each class?


#### Feature 5
- Genuines seems to follow a bimodal gaussian distribution with two modes around -1 and 1, while Counterfeits shows a trimodal distribution with peaks around -1, 0, 1.

- Therefore, there's some overlap in the ranges [-2, 0] and [0,2] while in [-0,5, 0.5] there are very few Genuine samples. 


#### Feature 6
- The situation is very similar to feature 5. This is indicative of the fact that Genuines are rare around 0 for both features (five and six), as confirmed by the scatter plot.

The scatter plot seems to show some clustering for each class when taking feature 5 and 6 jointly. In particular, a sample around 0 for both features is indicative of a counterfeit. Instead, samples around -1 and 1 can be distinguished looking at the value of the other feature. By looking at sample distances, it seems there are 4 clusters for each class.

