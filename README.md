# Polymer_Predict
NeurIPS - Open Polymer Prediction 2025 Kaggle Competition Submission:
I thought this particular dataset and task would be a very good application for contrastive learning, as similiar polymers would in turn have similar properties. I use rdkit to break down the SMILES strings and used a GNN encoder + constrastive learning to come up with an embedding for each polymer. Then checked to see if random forest classifiers or xgboost would result in a lower error. Can't share the data here due to kaggle policy but do search it up if you're interested!
