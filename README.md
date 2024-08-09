# SocialPulse-Predicting-Popularity-of-Posts-in-Social-Media-Platforms
## Introduction
Predicting the popularity of social media posts has long been a challenging task, with existing methods often falling short in incorporating the broader context of social network effects (SNE). Traditional models typically overlook crucial aspects such as group interactions, group popularity, and the count of group posts, which are instrumental in determining the reach and impact of a post within a social network. This project aims to address this limitation by developing an advanced prediction model that seamlessly integrates these novel social network effect parameters.

Our model leverages multi-channel metadata, encompassing image, text, numerical, and categorical data, to provide a comprehensive understanding of the factors influencing post popularity. By incorporating SNE context, we strive to enhance the accuracy and reliability of popularity predictions, offering deeper insights into the dynamics of social media interactions.

## Goals and Motivations
1. Improve Accuracy: Enhance the precision of predicting the popularity of social media posts by incorporating comprehensive SNE parameters.
2. Understand Influence: Explore and understand the impact of social networks on the popularity of social media posts.

## Approach
Our approach to enhancing the accuracy of social media post popularity prediction involves several key steps, from data integration to performance evaluation. Here’s a detailed breakdown:
### Data Integration
Integrate supplementary metadata with the SMPD 2019 dataset to enrich the data and provide a more comprehensive foundation for analysis.
### Model Architecture
Adapt the model architecture to accommodate changing input features and optimize activation functions for better performance and accuracy.
### Model Selection
Selected and employ a range of machine learning and deep learning models, including MobileNetV3Small, SBERT, Fusion Dense Neural Networks, Gradient Boosting, BiLSTM, LightGBM, CatBoost, and XGBoost; optimized hyperparameters for each model to enhance the accuracy of log-popularity prediction.
### Performance Evaluation
The models are evaluated using Mean Absolute Error (MAE) and Spearman Rank Correlation (SRC); and comparative analysis of two different frameworks to determine the most effective approach.

## About the Data
The SMPD 19 dataset, used in this project, comprises over 250,000 social media posts sourced from Flickr, covering the years 2015 and 2016. This extensive dataset is rich in diversity, including various data components such as numerical, categorical, image, text, and temporal-spatial information, providing a comprehensive basis for analyzing social media post popularity.
### Numerical Data
Features include ispro and canbuypro, along with engagement metrics such as post count and target attributes, offering quantitative measures of user interactions.
Categorical Data: Encompasses categories, subcategories, and concepts, which help classify and contextualize the posts.
### Images
Incorporates image data, allowing for visual analysis and exploration, enhancing the understanding of post content.
### Text Descriptions
Features such as titles, tags, and user descriptions provide rich textual context, adding depth to the analysis of post content and user engagement.
### Temporal-Spatial Information
Includes post and photo dates and spatial coordinates (latitude and longitude), capturing both the temporal and spatial dimensions of the data. The Post Date feature specifically captures when each post was made.
### Labels (Popularity Measurement)
Popularity scores are assigned based on engagement metrics, offering a quantitative measure of each post's popularity.
### Key Identifier Columns
The dataset includes a 'url' field containing the URL of each post, an 'img_file' field with the image file name, a 'Pid' (Post Identifier) uniquely identifying each post, and a 'Uid' (User Identifier) denoting the individual who made the post.
### Group Effect Features
Features like groups, group_impact, and group_weight provide insights into the impact of groups on post engagement. These features help understand the affiliations and influence of groups associated with each post, contributing to a nuanced analysis of social interactions and engagement within the Flickr platform.

## FRAMEWORK 1: Deep Learning Model Architecture
### Image Embeddings
The image data undergoes preprocessing, including resizing and normalization of pixel values based on mean and standard deviation. The MobileNetV3 Small model is employed to generate image embeddings. Batch processing is implemented to efficiently generate embeddings for a set of images, resulting in a concatenated array of embeddings. Subsequently, each individual image embedding is resized into a 2D array with a single sample and multiple features. 
### Text Embeddings
First, specific noise and year-related tags were removed. Then, a set of additional cleaning procedures are implemented, including the removal of punctuations, URLs, and emojis. Parallel processing was leveraged to the text column, applying a custom text preprocessing function to each text entry. The results of this preprocessing were stored in a new column named 'combined_text', containing the cleaned and processed textual data. Finally, the Sentence BERT was used to generate text embeddings for the preprocessed text. 
### Categorical Data
Categorical features, specifically 'Category', were transformed into a numerical format using one-hot encoding. The OneHotEncoder from scikit-learn was employed to convert the 'Category' column into a binary matrix. Additionally, the 'SubCategory' and 'Concept' columns underwent tokenization. This process converts textual data into numerical sequences, which are then padded to a maximum length of 1.
### Proposed Model
In this architecture, the image embeddings, text embeddings, encoded categorical data, and numerical data are concatenated and passed through a sequence of dense layers, each equipped with batch normalization and dropout for regularization. The model comprises four dense layers with ‘ReLU’ activation functions, which introduce non-linearity to the network. The final output layer, designed for regression tasks, utilizes a ‘linear’ activation function. The entire model is compiled using the Adam optimizer. The total number of trainable parameters is close to 1 million in this architecture. 
<img width="962" alt="Screenshot 2024-08-09 at 11 39 54 AM" src="https://github.com/user-attachments/assets/5fe8c56f-b443-406f-8fc5-2b1c8ed3edfc">

## Experimental Setup - Framework 1
### Predicting with the Social Network Effect (SNE):
It involves using evaluation metrics such as Mean Absolute Error (MAE) and Spearman Rank Correlation (SRC). The model starts with a dynamic learning rate and is initially set to train for 100 epochs, with early stopping invoked at the 90th epoch. The ReduceLROnPlateau scheduler adjusts the learning rate with a factor of 0.2 and a patience of 3, while EarlyStopping monitors validation MAE with a patience of 10. Performance metrics include SRC values of 0.7452 for training and 0.7262 for validation, and MAE values of 1.2442 for training and 1.2363 for validation. The model achieved a top 8 ranking on the leaderboard.
### Predicting without the Social Network Effec (SNE):
This also uses MAE and SRC as evaluation metrics, with a dynamic learning rate and an initial training of 100 epochs. Early stopping is invoked earlier, at the 38th epoch, and the ReduceLROnPlateau scheduler has the same parameters as above. Performance metrics for this model include SRC values of 0.7160 for training and 0.6939 for validation, and MAE values of 1.2967 for training and 1.2925 for validation, resulting in a top 21 ranking on the leaderboard.

## FRAMEWORK 2: Machine Learning Model Architecture
### Feature Engineering and Transformation
Before training the models, intrinsic features that are not directly available from the original dataset need to be extracted. This includes preprocessing original features, user metadata, and group metadata. Additionally, various features are extracted from different multimodal categories (images, text, categorical, spatio-temporal, numerical) to capture the full scope of multi-modality.
#### Images
Basic high-level features from each of the images such as image length, image width, total pixel count, and color mode are extracted. The color mode of the image includes {0: Grayscale 1: Palette 2: RGB 3: CMYK}. These extracted features are concatenated with the other features into a data frame. 
#### Spatio-temporal
From the existing feature datetime, new granular features such as hour, day, week_day, week_hour, year_weekday are created. Along with these, spatial features such as geoaccuracy, longitude, and latitude from the metadata are used. 
#### Textual features - GloVe embeddings
AllTags and Title columns containing tags and title of the posts as text are split into individual words and the word vectors are extracted using the pre-trained GloVe embeddings.  These are converted into word2vec format and then the extracted word vectors are averaged for each sentence. These averaged word vectors are used to depict the semantic information of AllTags and Title features. Furthermore, in order to represent the total number of words and characters in AllTags and Title features, Alltags_len, Alltags_number, Title_len, and Title_num are created by fetching the length and count of these two. 
#### Textual features - TF-IDF
To obtain more semantic and statistical information, TF-IDF representations of 'Title' and 'Alltags' columns are obtained (with n-gram in range (1,2)) and SVD is applied to reduce the dimensionality of the TF-IDF matrices into 20-dimensional feature sets. This helps in understanding the importance of each term with respect to the entire feature-set. 
#### Categorical features - Label Encoding
Categorical features including category, subcategory and concept are three-level hierarchical categories where categories consist of 11 classes, subcategory comprises 77 and concept comprises 668 classes. All these categorical features are converted into numerical format using Label Encoder. 
#### User Metadata
Using the Uid feature, the number of posts from each user is calculated and extracted into a new feature called Uid_count which represents the total number of posts by a particular user. Along with the extracted SNE features - group_weights, groups, and group_impact, user metadata information, all the above engineered features are concatenated and used to train the chosen machine learning algorithms CatBoost, XGBoost, and LightGBM.
### Proposed Models
In the study, the architectures of three gradient boosting models—CatBoost, LightGBM, and XGBoost—were utilized and compared. While these models share similarities in their gradient-boosting frameworks, their individual architectures and handling of data features exhibit distinct characteristics. All three models processed the data similarly at the outset. The data was split into training and submission sets with a stratified split based on the 'Category' feature. This approach ensures that both training and submission datasets are representative of the overall data distribution. 
#### CatBoost
The model employed a robust approach to handle categorical features, a standout feature of this framework. It automatically processed categorical variables like 'Category', 'Subcategory', 'Concept', etc., without the need for manual encoding. This capability significantly simplifies the preprocessing steps and potentially enhances the model's performance on categorical data. The model architecture was defined with a depth of 8 and utilized a gradient-based leaf estimation method. 
#### LightGBM 
The model did not require manual label encoding of categorical features. It operates on a different leaf-wise growth strategy compared to the level-wise growth of CatBoost and XGBoost, making it faster and more efficient, particularly on large datasets. LightGBM's framework was set up with 64 leaves and a maximum depth of 8, balancing the model's complexity and computational efficiency. 
#### XGBoost
XGBoost, on the other hand, required pre-encoded categorical features. Similar to CatBoost, it used a maximum depth of 8 in its tree structure. It's highly customizable, allowing for in-depth parameter tuning, but this also means it might require more manual configuration compared to CatBoost.
<img width="889" alt="Screenshot 2024-08-09 at 11 40 06 AM" src="https://github.com/user-attachments/assets/3e0719ac-de8e-4fcd-970d-6a0e04c69c2b">

## Experimental Setup - Framework 2
### CatBoost
The objective is to minimize RMSE with MAE as the evaluation metric. The learning rate is set at 0.03, with an L2 leaf regularization parameter of 3, max CTR complexity of 1, and a maximum depth of 8. The leaf estimation method used is gradient descent, and the model is trained for up to 10,000 iterations with early stopping triggered after 50 rounds without improvement.
### XGBoost 
The objective is regression with squared error, and MAE is used for evaluation. The learning rate is 0.03, the maximum depth is 8, and the model is trained for 10,000 iterations, also employing early stopping after 50 rounds without improvement.
### LightGBM
The aim was regression with MAE as the metric. It uses a learning rate of 0.03, with 64 leaves, a maximum depth of 8, and is trained for 1

## Combined Results for Frameworks 1 & 2
The performance of different models was evaluated using Mean Absolute Error (MAE) and Spearman's Rho (SRC). The CatBoost model achieved an MAE of 0.5898 and an SRC of 0.9226. LightGBM outperformed the other models with an MAE of 0.5451 and an SRC of 0.9259. The XGBoost model recorded an MAE of 0.5616 and an SRC of 0.9222. In comparison, the Multimodal Deep Learning (DL) model had a higher MAE of 1.25 and a lower SRC of 0.721, indicating less accurate predictions than the tree-based models.
<img width="743" alt="Screenshot 2024-08-09 at 11 40 18 AM" src="https://github.com/user-attachments/assets/2a00de6b-c60b-481e-ae62-b52e9a02cd57">

## Conclusion
The results of this study confirm that visibility in social networks significantly impacts the accuracy of post popularity prediction models. Our findings highlight the necessity of engineering more social network features to enhance model performance further. Through sophisticated feature engineering using NLP and ML techniques, we achieved a top rank on the leaderboard with Framework-2. Meanwhile, our Deep Learning Framework-1 secured a position in the top 10.
An essential insight from our research is that when dealing with image and text information, “less is more.” Providing more information does not necessarily improve the model’s performance; instead, incorporating the right information is crucial. Achieving this balance requires a deep understanding of the domain.

## Future Work
Future efforts could focus on fine-tuning the embedding models used for image and text data within the context of social media to improve their relevance and accuracy. Experimenting with different image and text embedding models may also uncover better approaches suited for social media data. Additionally, enhanced feature engineering is necessary to continuously integrate new social network features that capture the nuances of user interactions and network dynamics. Developing advanced techniques for multimodal data fusion will further leverage the strengths of each data type, leading to more accurate predictions and deeper insights into social media post popularity.


