# Do Wars affect a Story’s Emotional Arc? Books as a Collective Coping Mechanism

TLDR - We did this: [Kurt Vonnegut on the Shapes of Stories](https://www.youtube.com/watch?v=oP3c1h8v2ZQ&t=186s)

This project was done in collaboration with [Eric Frey](https://www.linkedin.com/in/eric-frey-a27b27114/) and [Davis Thomas](https://www.linkedin.com/in/davis-jacob-thomas/) at the Barcelona School of Economics. 

After stumbling upon [Project Gutenberg](https://www.gutenberg.org/), a database of 60,000 ebooks we were curious to uncover if we could extract the emotional arc of a book and determine if there was commonality of certain arc shapes across other books. Our program director then asked us to take it a step further and tie the frequency of emotional arcs to a societal shock. Therefore for this project, we use World War I (WWI) as an example emotional shock to society and compare the frequency of the emotional arcs shapes in the books published leading up to the war and after it. 

The steps done in order to complete this project were as follows:
1. Use webscraping techniques to obtain a list of book titles and authors for stories written between 1890s to the 1930s. This results in a list of 1,200 books.
2. Query the [Gutendex API](https://gutendex.com/) with this list to obtain the text files of the books. Unfortunately, we could only download 139 ebooks. This may be due to the books not being in the public domain. Project Gutenberg only publishes books where the U.S. copyright has expired, which can be typically 70 years after the author’s death.
3. Use Natural Language Processing techniques to assemble a clean corupus of text, tokenize it, and run sentiment analysis in order to retrieve the emotional arc per book
4. Use Singular Value Decomposition (SVD) to isolate commonality from the amassed emotional arcs 
5. Use K-means clustering on the normalized mode coefficients obtained through (SVD) to cluster the different books based on the emotional arc that best describes the book
6. Obtain the change in frequency of the uncovered emotional arcs before and after WWI and using a chi-squared hypothesis test see if this change is statistically significant.

Based on the results of the chi-squared test, we can not state with certainty that WWI had an impact on the frequency of books associated with the uncovered emotional arcs. The p-value of the test was .804 which is greater than our desired significance level of .05.

Below are figures associated with the emotional arcs we uncovered in the books we used:


<img width="873" alt="Screenshot 2023-08-10 at 8 20 04 PM" src="https://github.com/erikaguti/Project-Gutenberg/assets/57955273/911adb63-f13e-46f5-b59a-96baf3d6c374">


<img width="904" alt="Screenshot 2023-08-10 at 8 20 45 PM" src="https://github.com/erikaguti/Project-Gutenberg/assets/57955273/c36ccbf7-d480-40d5-a4ef-eed48a4bb6a8">


The write-up is [here](https://github.com/erikaguti/Project-Gutenberg/blob/main/Project_Gutenberg.pdf). Although our results were inconclusive, our analysis provides a foundation towards contributing to the understanding of how collective traumas can shape cultural production and the role of literature as a coping mechanism for individuals during difficult times.


