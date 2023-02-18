# Youtube-Comment-Spam-Detection
This project aims to develop a machine learning algorithm that can accurately detect and filter out spam comments on YouTube videos.

## Overview
YouTube is one of the most popular video-sharing platforms in the world, with millions of videos uploaded and billions of views generated every day. Unfortunately, this popularity has also attracted spammers, who use the platform to post irrelevant or promotional comments that can detract from the overall user experience.

To address this problem, we have developed a machine learning algorithm that can automatically identify and flag spam comments on YouTube videos. The algorithm is based on a combination of Bernoulli's Naive Bayes Algorithm and statistical analysis, and has been trained on a large dataset of labeled comments.

## Dataset
The dataset used for training and testing the algorithm is a collection of YouTube comments that have been labeled as either 1(spam) or 0(legitimate). The dataset is relatively small with approx 2000 tuples.

## Requirements
The following software and libraries are required to run the code:

1. Python 3.7 or higher
2. Pandas
3. NumPy
4. Scikit-learn

## Usage
To use the algorithm, simply run the youtube_spam_detection.py script and provide the path to a CSV file containing the comments you want to analyze.Then the script will run and will ask user to enter a random comment which it will predict whether its a spam or not.

## Conclusion
Overall, this machine learning algorithm has demonstrated high accuracy in detecting spam comments on YouTube videos. This can help improve the overall user experience and ensure that the platform remains a valuable resource for sharing and discovering new content
