# DILS_2018

### Research Question: Can we distinguish infected subjects from non-infected subjects at an early stage of a disease using gene expression profiles?  

We used four well-established Machine Learning algorithms to predict the state of respiratory viral infection at an early stage by analysing gene expression data. We performed experiments considering two early time-points; at nearest to 48 hours and at nearest to Onset time. Onset time is the time when a person starts showing symptoms of an infection. We establish that the prediction at an early stage is possible with considerable accuracy, 82.83% accuracy at nearest to 48 hours and 85.45% accuracy at nearest to Onset time using 10-fold cross-validation, and accuracies of 80% and 82.14%, respectively, on the hold-out test set. Based on performed t-tests, we can conclude that there is no significant difference between accuracy obtained at nearest to 48 hours and accuracy obtained at nearest to Onset time which suggests that it is possible to distinguish infected subjects from non-infected subjects even before the subjects start showing symptoms of an infection. Moreover, we have identified top 10 most important genes which are having the maximum contribution in the progression of the respiratory viral infection at the early stage. The diagnosis and prevention of the respiratory viral infection at the early stage by targeting these genes can potentially improve the results than targeting the genes affected at the later stage of the infection. The four Machine Learning algorithms used for this study are k-Nearest Neighbour, Random Forest, Linear Support Vector Machine, and Non-Linear Support Vector Machine with RBF kernel. 

### The labelled gene expression data sets and codes are available on this GitHub repository. 

This work is published in DILS 2018 Conference. Link to the published paper: https://link.springer.com/chapter/10.1007/978-3-030-06016-9_11

## How to Install and Run the Software
The code is written in R programming language and can run on Windows, MacOS, or Linux. However, the R programming language is required to be installed before you run the software/code.

### Required Programming Language:
R version 3.6.2 or above

You can download the latest version of R from here:
* [Install R](https://www.r-project.org/)

### Required packages:
Install all the packages listed in the requirements.txt file

### Steps to run the code:
1. Download the provided R code and gene expression datasets, and keep them in the same folder. 
2. Open the terminal.
3. Go to the folder where you downloaded all the codes and datasets. You can use cd command for that.
4. Run the code using this command: 
```
R CMD BATCH code_file_name.R
```
These commands will create a .Rout (output) file in the same folder. This .Rout file will contain all the results. 
