# GA-EM
A genetic version of the EM algorithm for Gaussian Mixture Models.

`GA_EM_main.m` is set up to generate data using `SampleData.m` and then run the EM and GA-EM algorithms on that data set.

If you have a different data set, you can either modify `GA_EM_main.m` to use this data instead of the sample data or write your own main function and call `GA_EM(data)`. The input data should be an `N x d` matrix where `N` is the number of observations, and `d` is the dimension of the feature space.

**Configuration:**
- In `GA_EM_main.m`, you can set the number of points, components, and dimensions (`N, C, d`) to generate sample data.
- In `GA_EM.m`, you can set most of the genetic algorithm parameters, except the thresholds described below.
- In `Recombine.m`, set the threshold for eliminating unsupported components.
- In `Enforce.m`, set the maximum allowed correlation between components.
