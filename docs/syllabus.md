# Syllabus

Python crash course
---
1. <details><summary>Jan 13 T - Intro and python basics 1</summary>

     - You will get a brief overview of the course.
     - You will be able to use `conda` and `pip` to manage python environments.
     - You will be able to run python code in a `.py` file or `.ipynb` Jupyter notebook file.
     - You will understand some basic python syntax.

    </details>

2. <details><summary>Jan 15 R - Python basics 2</summary>

     - You will understand some more basic python syntax.

    </details>

3. <details><summary>Jan 20 T - Python basics 3</summary>

    - You will understand all the basic python syntax needed to start using python for real projects.

    </details>

Working with data
---
4. <details><summary>Jan 22 R - Data arrays and plots</summary>

     - You will appreciate that many types of data can be represented as N-dimensional arrays.
     - You will understand how to work with `numpy` N-dimensional arrays.
     - You will appreciate that `numpy` can be *much much* faster than raw python.
     - You will appreciate that without `numpy` we would not use python for most data analysis.
     - You will be able to visualize data with simple plots using `matplotlib`.

    </details>
    
5. <details><summary>Jan 27 T - Random walk lab</summary>

     - You will write a program to simulate random walks in various dimensions and with different types of boundaries.

    </details>

6. <details><summary>Jan 29 R - Tabular datasets</summary>

     - You will be able to to work with tabular data sets using `pandas`.
     - You will be able to use `seaborn` and `hvplot` to create plots from a `pandas` dataframe.
     - You will appreciate how useful `pandas` is for exploratory data analysis.

    </details>

Probability and random variation
---
7. <details><summary>Feb 3 T - Probability distributions</summary>

     - You will understand the difference between a probability and a probability density.
     - You will understand how some basic probability distributions relate to distinct types of random behavior.
     - You will be able to visualize how well a probability distribution explains data.
     - You will be able to use a probability distribution to make probabalistic predictions.

    </details>

8. <details><summary>Feb 5 R - Resampling</summary>

     - You will understand the difference between a population distribution and a sampling distributuion.
     - You will appreciate that statistics for different samples are likely to vary.
     - You will understand the concept of a confidence interval.
     - You will be able to compute a confidence interval using bootstrapping.
     - You will be able to test the hypothesis that two samples come from the same population distribution using a permutation test.
     - You will appreciate how the Central Limit Theorem explains why normal-ish distributions are frequently observed in biological measurements.

    </details>

Model optimization
---
9. <details><summary>Feb 10 T - Curve fitting and maximum likelihood estimation (MLE)</summary>

     - You will be able to fit a function to data by minimizing the residuals.
     - You will be able to fit an arbitrary probability distribution to data by maximizing the loglikelihood.
     - You will understand the concept of gradient descent minimization.
     - You will appreciate the difference between local and global optimization.

    </details>

Machine learning
---
10. <details><summary>Feb 12 R - Linear regression</summary>

     - You will be able to fit a line to X vs. Y data.
      - You will be able to fit a (hyper-)plane to {X0, X1, X2, ...} vs. Y data.
      - You will be able to quantify your model's goodness of fit and use the model to make predictions.
      - You will be able to compute confidence intervals for all model parameters and visualize a confidence envelope for your fit.
      - You will appreciate why data points with high leverage can greatly influence your fit.
      - You will understand under what conditions you may want to standardize your features {X0, X1, X2, ...}.
      - You will appreciate the benefits of a simple linear model: fast computations due to analytical solution and straightforward interpretation of model parameters.

    </details>

11. <details><summary>Feb 17 T - Nonlinear regression</summary>

     - You will be able to use a polynomial or K-nearest neighbors (KNN) model to explain the dependence of a target variable Y on feature variables {X}.
      - You will understand how polynomial regression can be recast as a simple linear regression.
      - You will appreciate that although a KNN model can be used to explain or predict lots of arbitrary nonlinear relations, it is less obvious what the model means.

    </details>

12. <details><summary>Feb 19 R - EXAM 1</summary>

     - Cumulative exam for all topics covered up to this point.

    </details>

13. <details><summary>Feb 24 T - Cross validation</summary>

     - You will be able to split your dataset up into training and testing sets.
      - You will understand the difference between training error and testing error.
      - You will appreciate that often the best model is the one that will generalize best to new data (i.e., has the lowest testing error, not the lowest training error).
      - You will understand the concept of the "bias vs. variance" tradeoff.
      - You will be able to perform K-fold cross validation to select the model that fits well without overfitting noise.

    </details>

14. <details><summary>Feb 26 R - Regularization</summary>

     - You will appreciate how correlations can influence a linear regression.
      - You will be able to perform ridge and lasso regression.
      - You will appreciate how regularization can prevent poorly constrained model parameters from exploding.
      - You will appreciate how lasso regularization can identify model parameters with little to no impact.
      - You will understand how to choose (tune) the regularization hyperparameter.

    </details>

15. <details><summary>Mar 3 T - Generalized linear model (GLM)</summary>

     - You will gain a conceptual understanding for a generalized linear model (GLM).
      - You will appreciate why a GLM may be a better choice than a simple linear model for neural spiking data.
      - You will use a GLM (Poisson regression) to predict a neuron's spiking in response to a stimulus.
      - You will see how the choice of noise distribution in a GLM can be used for binary classification.
      - You will use a GLM (logistic regression) to predict a mouse's left vs. right choice from its neural activity.

    </details>

16. <details><summary>Mar 5 R - Classification</summary>

     - You will understand that classification involves modeling the categorical grouping of data.
      - You will be able to use a logistic regression binary classifier.
      - You will be able to use your classifier to predict the class to which data belongs.
      - You will be able to compute the accuracy of your classifier given data with known class labels.
      - You will be able to use your classifier to get the probability of each possible class.
      - You will be able to compute cross validated predictions, accuracy, and probabilities.
      - You will be able to generate a confusion matrix for your classifier.
      - You will be able to generate a ROC curve for your classifier.
      - You will gain a conceptual understanding for classification with a support vector machine (SVM).
      - You will be able to use a SVM classifier to separate data with linear boundaries.
      - You will appreciate at the conceptual level that SVM can achieve complex nonlinear boundaries by projecting the data into higher dimensions.
      - You will be able to use a SVM classifier to separate data with nonlinear boundaries.

    </details>

17. <details><summary>Mar 10 T - Clustering</summary>

     - You will appreciate the difference between classification and clustering (i.e., no labels to train on).
      - You will understand and be able to use several different clustering algorithms to segregate data.
      - You will appreciate that each clustering algorithm has its own pros and cons.
      - You will be able to use several different empirical metrics to choose an optimal clustering model (e.g., number of clusters).
      - You will use the Bayesian information criterion (BIC) to choose the optimal number of clusters for a gaussian mixture model (GMM).

    </details>

18. <details><summary>Mar 12 R - Dimensionality reduction</summary>

     - You will understand the concept of principal component analysis (PCA).
      - You will be able to interpret the principal components as directions in the original data space.
      - You will be able to quantify the amount of variance explained by any given number of principal components.
      - You will understand how series can be represented as points in a high dimensional space.
      - You will be able to apply PCA to images.
      - You will see how PCA can be used as a filter to remove noise.

    </details>

19. <details><summary>Mar 24 T - EXAM 2</summary>

     - Cumulative exam for all topics covered up to this point, but will focus more on the topics covered after exam 1.

    </details>

20. <details><summary>Mar 26 R - EEG and RNAseq datasets</summary>

     - You will apply PCA to EEG time series.
      - You will be able to cluster time series and visualize the clustering in a low number of PCs.
      - You will appreciate how clustering of time series could be beneficial for interpreting experimental data.
      - You will walk through an example of clustering in reduced dimensions for single cell RNAseq data.
      - You will appreciate the importance of being able to think critically about your data.

    </details>

Modeling time series
---
21. <details><summary>Mar 31 T - Data series and convolution</summary>

     - You will appreciate that data points in sequences are correlated (unless pure noise) as opposed to independent random variables.
     - You will appreciate how undersampling can introduces aliasing artifacts in a sequence.
     - You will be able to visualize the frequency power spectrum of a 1-D sequence.
     - You will be able to visualize the frequency spectrogram of a 1-D sequence.
     - You will understand why convolution describes a systems output based on its impulse response.
     - You will be able to convolve two 1-D sequences.
     - You will appreciate how convolution can be used to filter a sequence.
     - You will be able to apply lowpass, highpass and bandpass finite impulse response (FIR) filters to a 1-D sequence.
     - You will be able to properly downsample a 1-D sequence without introducing aliasing artifacts.
     - You will be able to convolve two 2-D sequences (e.g., images).
     - You will appreciate that convolution can be used to highlight features in an image.
     - You will appreciate that the joint probability distribution resulting from adding two random variables is the convolution of their individual probability distributions.

    </details>

22. <details><summary>Apr 2 R - Leaky integrate and fire (LIF) neuron</summary>

     - You will appreciate how a cell membrane can be approximated by a simple RC circuit.
      - You will understand the concept of the LIF neuron model.
      - You will be able to simulate a LIF neuron.
      - You will be able to plot spike rasters.

    </details>

23. <details><summary>Apr 7 T - LIF neuron with synaptic input</summary>

     - You will be able to simulate stochastic synaptic input to a LIF neuron.
      - You will appreciate how convolution can be used to integrate synaptic inputs.

    </details>

24. <details><summary>Apr 9 R - Hidden Markov model (HMM) 1</summary>

     - You will understand the concept of a hidden Markov model (HMM).
      - You will use an HMM to model current flowing through a single ion channel.
      - Given an HMM, you will be able to compute the most likely state trajectory for a data sequence.
      - You will appreciate how an HMM uses the full sequence to inform the model.
      - You will use the Bayesian information criterion (BIC) to choose the best model out of several possibilities.

    </details>

25. <details><summary>Apr 14 T - Hidden Markov model (HMM) 2</summary>

     - You will use an HMM to predict exons and introns in a nucleotide sequence.

    </details>

Neural networks
---
26. <details><summary>Apr 16 R - Feedforward neural network (FNN)</summary>

     - You will understand the basic concept of a neural network as a universal function generator.
     - You will understand how the input and output layers of a neural network depend on the data and desired computation.
     - You will understand the concept of how a neural network is trained.
     - You will be able to implement basic feed-forward neural networks for regression and classification in Python.
     - You will appreciate that neural networks are not always the best choice.

    </details>

27. <details><summary>Apr 21 T - Convolutional neural network (CNN)</summary>

     - You will be able to implement neural networks using PyTorch.
     - You will understand the basic concept of a CNN.
     - You will apply a CNN to decipher grating orientations based on images of gratings.

    </details>

28. <details><summary>Apr 23 R - EXAM 3</summary>

     - Cumulative exam for the entire semester, but will focus more on the topics covered after exam 2.

    </details>
