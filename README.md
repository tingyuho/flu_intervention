##flu_vaccination ##

the mathematical and computer models of the spread of infectious disease which is modified based on [FluTE](https://www.cs.unm.edu/~dlchao/flute/) model but given the two insurance-based incentives:

1.	vaccination reimbursement
2.	outpatient cost-sharing rate

Two low-fidelity model builder are built in this study:
1.	Active learning approach - symbolic regressor
2.	Machine learning approach - random forest regressor

The process breaks down into:

- config file is used for the setting of infectious process ex: reimbursement 10 & costsharingrate 0.1 (the patients pay 10% of medical cost)
- launch mingw command prompt and run mingw32-make
- run the model: flute config-*

Output file
- insurer_cost_summary Active_learning_based_PBnB applied active learning like symbolic regression and machine learning like random forest regressor to learn the intervention-cost relationship, which is black box for the decision maker
