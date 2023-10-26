# Welcome to: Predicting anti-inflammatory peptides by ensemble machine learning and deep learning
Inflammation is a biological response to harmful stimuli, aiding in maintaining tissue homeostasis. However, excessive or persistent inflammation can precipitate a myriad of pathological conditions. Although current treatments like NSAIDs, corticosteroids, and immunosuppressants are effective, they can have side effects and resistance issues. In this backdrop, anti-inflammatory peptides (AIPs) have emerged as a promising therapeutic approach against inflammation. Leveraging machine learning methods, we have the opportunity to accelerate the discovery and investigation of these AIPs more effectively. In this study, we proposed an advanced framework by ensemble machine learning and deep learning for AIP prediction. Initially, we constructed three individual models with extremely randomized trees (ET), gated recurrent units (GRU), and convolutional neural networks (CNN) with attention mechanism, and then used stacking architecture to build the final predictor. By utilizing various sequence encodings and combining the strengths of different algorithms, our predictor has demonstrated exemplary performance. On our independent test set, our model achieved an accuracy, MCC, and F1-score of 0.757, 0.500, and 0.707, respectively, clearly outperforming other contemporary AIP prediction methods. Additionally, our model offers profound insights into the feature interpretation of AIPs, establishing a valuable knowledge foundation for the design and development of future anti-inflammatory strategies.

This AIP prediction tool developed by a team from the Chinese University of Hong Kong (Shenzhen)

![The workflow of this study]([main/workflow.jpg])


# Dataset for this study
We provided our dataset and you can find them [Dataset](https://github.com/GGCL7/AIP_MDL/tree/main/Dataset)

# Feature generator
We provided our code and you can find them [Feature generator](https://github.com/GGCL7/AIP_MDL/tree/main/Feature%20generator)

# Use our model
We developed three models, and you can find the pre-stored models at [Model](https://github.com/GGCL7/AIP_MDL/tree/main/Model).
