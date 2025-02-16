## Logistics Regression

### Acknowledgement - Ken Church  Coursera

We would like to thank Ken Church for lending his expertise to review the outline and topics of this specialization.

[Ken ChurchOpens in a new tab](https://scholar.google.com/citations?user=E6aqGvYAAAAJ) is currently a Distinguished Scientist at Baidu, and formerly led computational linguistics research at IBM, Microsoft, AT&T and Johns Hopkins University.  He was the president of the Association for Computational Linguistics (ACL) in 2012, and also president of the Special Interest Group on Linguistic Data & Corpus-based Approaches to Natural Language Processing (SIGDAT) from 1993 until 2011.

Thanks Ken for sharing your insights on what learners should know about natural language processing!

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/m3ryXNYeT7K68lzWHu-ylQ_8820db2bd7fb7d1f2c7324f612d06bf3_ken_church_small.png?expiry=1737849600000&hmac=qVByQZtF27bkCFBMWFI7kUw3KuBOgcb3NUBXFlk_qG0)



### Supervised ML & Sentiment Analysis

In supervised machine learning, you usually have an input XX, which goes into your prediction function to get your Y^\\hat Y. You can then compare your prediction with the true value YY. This gives you your cost which you use to update the parameters θ\\theta. The following image, summarizes the process.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/3vVQr0zhTlW1UK9M4V5Vww_e5b656e0d36041e1be5edd54bd3562a5_Screen-Shot-2020-09-01-at-7.35.25-AM.png?expiry=1737849600000&hmac=VuQx543vSsK23eIFEsLQGPJ2vc6q7IgLPPTktDqGmdo)

To perform sentiment analysis on a tweet, you first have to represent the text (i.e. "I am happy because I am learning NLP ") as features, you then train your logistic regression classifier, and then you can use it to classify the text.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/z84yBRbCS0qOMgUWwktKdA_927d67ce246c4212ad9d389a0243ed0e_Screen-Shot-2020-09-01-at-7.41.26-AM.png?expiry=1737849600000&hmac=7Lzrti2HYe7rMPO4WPVViXMaU30bJm9UU7YLmor0BNU)

Note that in this case, you either classify 1, for a positive sentiment, or 0, for a negative sentiment.



### Vocabulary & Feature Extraction

Given a tweet, or some text, you can represent it as a vector of dimension VV, where VV corresponds to your vocabulary size. If you had the tweet "I am happy because I am learning NLP", then you would put a 1 in the corresponding index for any word in the tweet, and a 0 otherwise.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/vpVZPKHCS6uVWTyhwmurrQ_d2e2fd874a354ab38047ef531021b681_Screen-Shot-2020-09-01-at-7.48.25-AM.png?expiry=1737849600000&hmac=Dv00Om4yE05ilV_Cq6Bpa1OsWU5gX2Yq5-a9KBe2vPU)

As you can see, as VV gets larger, the vector becomes more sparse. Furthermore, we end up having many more features and end up training θ\\theta VV parameters. This could result in larger training time, and large prediction time.



### Feature Extraction with Frequencies

Given a corpus with positive and negative tweets as follows:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/zhNAjggWTbWTQI4IFu21ZA_32f86a38bc224959be21ede407128c82_Screen-Shot-2020-09-01-at-7.55.08-AM.png?expiry=1737849600000&hmac=9m_XKnXl9YVAPqDNQb2SLmJ7zWhBfR81ylLH3I7KhqQ)

You have to encode each tweet as a vector. Previously, this vector was of dimension VV. Now, as you will see in the upcoming videos, you will represent it with a vector of dimension 33. To do so, you have to create a dictionary to map the word, and the class it appeared in (positive or negative) to the number of times that word appeared in its corresponding class.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/vhHO7A7dTvuRzuwO3U779Q_5364f83a7bd54782a09279efe06e96f2_Screen-Shot-2020-09-01-at-7.57.30-AM.png?expiry=1737849600000&hmac=QsmlXaSQByTJirvPdAslhfTAdN-kDAr3BkHcDVrmqHc)

In the past two videos, we call this dictionary \`freqs\`. In the table above, you can see how words like happy and sad tend to take clear sides, while other words like "I, am" tend to be more neutral. Given this dictionary and the tweet, "I am sad, I am not learning NLP", you can create a vector corresponding to the feature as follows:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N_PzQkNvSNqz80JDb-ja5A_a44a87942c5e476593cd8e1582cf5b06_Screen-Shot-2020-09-01-at-8.04.10-AM.png?expiry=1737849600000&hmac=Qsbzb-euBJkBLfEXS4utolaTLy8L9ngrT9FRFangDYI)

To encode the negative feature, you can do the same thing.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dQU0vX1NT3yFNL19TS98gQ_b885bb87a6d644389726ddc740869153_Screen-Shot-2020-09-01-at-8.04.21-AM.png?expiry=1737849600000&hmac=W1S1ik9Ug7ZnzFHVasXdacoXSAuVsQtZhuZzfYo8rXM)

Hence you end up getting the following feature vector \[1,8,11\]\[1,8,11\]. 11 corresponds to the bias, 88 the positive feature, and 1111 the negative feature.


### Preprocessing

When preprocessing, you have to perform the following:

1.  Eliminate handles and URLs
    
2.  Tokenize the string into words.
    
3.  Remove stop words like "and, is, a, on, etc."
    
4.  Stemming- or convert every word to its stem. Like dancer, dancing, danced, becomes 'danc'. You can use porter stemmer to take care of this.
    
5.  Convert all your words to lower case.
    

For example the following tweet "@YMourri and @AndrewYNg are tuning a GREAT AI model at https://deeplearning.ai!!!" after preprocessing becomes

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dKnqOOtTQP6p6jjrU2D-LQ_3e6a23c4313b481f9301bbba086b6f66_Screen-Shot-2020-09-01-at-8.14.55-AM.png?expiry=1737849600000&hmac=JAfLlln7fxmIU0JgsHLli3Iuoj1aTvhxs017bNgrxsU)

\[tun,great,ai,model\]\[tun, great, ai, model\]. Hence you can see how we eliminated handles, tokenized it into words, removed stop words, performed stemming, and converted everything to lower case.



### Putting it all together

Over all , you start with a given text, you perform preprocessing, then you do feature extraction to convert text into numerical representation as follows:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_OTVrbipS9ek1a24qXvXFw_641fc5c72d974a4582c229f6ebf073f5_Screen-Shot-2020-09-01-at-8.20.08-AM.png?expiry=1738022400000&hmac=vFWeDdNBNgsjkUOCIaA7PJjfSuKbZVFeytbnf49apog)

Your XX becomes of dimension (m,3)(m,3) as follows.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/C3BC-AlDRj-wQvgJQyY_8w_6b189b0ef72e456b9cce8c264796f567_Screen-Shot-2020-09-01-at-8.24.17-AM.png?expiry=1738022400000&hmac=IM6tltcZW2_T1Jb6w0V7h-FCjsscfiIjIjv-dHD-YjE)

When implementing it with code, it becomes as follows:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/NwmjyOjKQTaJo8joyqE2Ow_330a68fd246c44c3a5409216b096559c_Screen-Shot-2020-09-01-at-8.20.48-AM.png?expiry=1738022400000&hmac=-CGwtCBEe0ICD0x41ti-3HYsC3li5e2MV-r0TEX3wmM)

You can see in the last step you are storing the extracted features as rows in your XX matrix and you have mm of these examples.



### Logistic Regression Overview

Logistic regression makes use of the sigmoid function which outputs a probability between 0 and 1. The sigmoid function with some weight parameter θ\\theta and some input x(i)x^{(i)} is defined as follows.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/oL4Ox_JxTBi-DsfycUwYvw_d0582a0dddf7470486f0955c8b025dd6_Screen-Shot-2020-09-01-at-8.30.00-AM.png?expiry=1738108800000&hmac=1cvr-EmF6uH0qSpopL0PaQR9MvZtd_Cc7_ZXbt72DLw)

Note that as θTx(i)\\theta^Tx^{(i)} gets closer and closer to −∞\-\\infty the denominator of the sigmoid function gets larger and larger and as a result, the sigmoid gets closer to 00. On the other hand, as θTx(i)\\theta^Tx^{(i)} gets closer and closer to ∞\\infty the denominator of the sigmoid function gets closer to 1 and as a result the sigmoid also gets closer to 11.

Now given a tweet, you can transform it into a vector and run it through your sigmoid function to get a prediction as follows:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/THV0BbogT2i1dAW6IA9oMg_67bcc86617b54ac4b5750d51d032cd8f_Screen-Shot-2020-09-01-at-8.37.07-AM.png?expiry=1738108800000&hmac=0_7loLAyNwYrNTeLECrsUOFml97G6BsyLbmyeZ2QS48)



### Logistic Regression: Training

To train your logistic regression function, you will do the following:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/YGmjEyR0Sw2poxMkdBsNeQ_74cb9a1075fb4d1eb835b14a8d5b2456_Screen-Shot-2020-09-01-at-8.39.39-AM.png?expiry=1738108800000&hmac=cfq56zix-z3E_uKsKOlk8R4Atum7ZRJSpHK9ftVhdQE)

You initialize your parameter θ\\theta, that you can use in your sigmoid, you then compute the gradient that you will use to update θ\\theta, and then calculate the cost. You keep doing so until good enough.

**Note:** If you do not know what a gradient is, don't worry about it. I will show you what it is at then end of this week in an optional reading. In a nutshell, the gradient allows you to learn what θ\\theta is so that you can predict your tweet sentiment accurately.

Usually you keep training until the cost converges. If you were to plot the number of iterations versus the cost, you should see something like this:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/kj9HzV22Qp2_R81dttKdLw_97bb4671b7384b0c8b8dbf59b253dea1_Screen-Shot-2020-09-01-at-8.49.51-AM.png?expiry=1738108800000&hmac=a8e3Q5u0ODgjpbsKC73_AgrKGxaEUkYZ8ozxT9kC-2s)