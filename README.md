# Smart Firewall for Network Communication and Policy Prediction

This project was a part of our prototype built for the [2018 Viterbi Graduate Hackathon](http://www.csclass.info/USC/18Hackathon/)

## Solution Proposed

Data collected from various sources run on different machine learning models to build an ensemble that judges the transaction to be a good one or bad. 

Apply further policies to decide if we block the transaction or allow it to continue.

### Data generation

    Collected data from multiple sources available online - packet data (eg. LAN traffic from a source in Portland Oregon)
    Generated real time data using active listening techniques with Wireshark and vWave to assess network and packet information for device identification

### Data cleaning

    Realized that port number in a certain range are not so common as compared to others. 
    Created hash maps of port numbers, MAC addresses, IP addresses and resolved hosts along with Network Protocol
    Normalized data as and where required
    
### Model building

#### KMeans clustering:
      Mapped data into two clusters and tested accuracy of model against known results.
      Accuracy of 65% was achieved and was used as a baseline to improve prediction schemes
      
#### Logistic regression:
      Applied simple logistic regression to the improved dataset 
      Achieved 85.3% F1 score and 87.5% precision
      
#### XGBoost:
      Boosted trees for policy identification and modeling
      Achieved 86.2% F1 score and 86.9% precision
    
#### Neural Networks:
      Built a neural network with 2 hidden layers and 12,6 nodes each.
      Applied 160 iterations
      Achieved 87.8% F1 score and 88.0% precision
      
We tested these models individually on a test set with 650,000+ transactions of which there were 4 known amlicious packets to be identified.

### Results

Results of each model are as follows:-
    
  |     Model    | Count of Bad Packets detected | Count of Bad packets blocked |
  |     :---:    |            :---:              |              :---:           |
  |   K Means    |             1200              |                 1            |
  | Logistic Regression  | 737  | 3 |
  | XGBoost | 201 | 2  |
  | Neural Networks | 3 | 1 |
  
We see that as the model is being trained with different machine learning algorithms and techniques, the number of false positives decrease remarkably and close to accurate detection of good and bad packets is made.
