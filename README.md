# Polymer_Predict
NeurIPS - Open Polymer Prediction 2025 Kaggle Competition Submission:
I thought this particular daatset and task would be a very good application for contrastive learning, as similiar polymers would in turn have similar properties. I use rdkit to break down the SMILES values and used a GNN encoder + constrastive learning to come up with an embedding for each polymer. Then checked to see if random forest classifiers or xgboost would classify the values better. Can't share the data here due to kaggle policy but do search it up if you're interested!
