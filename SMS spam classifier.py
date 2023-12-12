#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as nd
import pandas as pd


# In[3]:


df = pd.read_csv('spam.csv.zip')


# In[4]:


df.sample(5)


# In[5]:


df.shape


# In[6]:


# 1. Data cleaning
# 2. EDA
# 3. Text preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. website
# 8. Deploy


# ## 1.Data Cleaning

# In[7]:


df.info()


# In[11]:


df.sample(5)


# In[13]:


# renaming the cols
df.rename(columns={'v1':'label','v2':'text'},inplace=True)
df.sample(5)


# In[15]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[16]:


df['label'] = encoder.fit_transform(df['label'])


# In[17]:


df.head()


# In[18]:


# missing values
df.isnull().sum()


# In[19]:


# check for duplicate values
df.duplicated().sum()


# In[20]:


df.shape


# ## 2.EDA

# In[21]:


df.head()


# In[23]:


df['label'].value_counts()


# In[25]:


import matplotlib.pyplot as plt
plt.pie(df['label'].value_counts(),labels=['x','spam'],autopct="%0.2f")


# In[26]:


# Data is imbalanced


# In[27]:


import nltk


# In[29]:


get_ipython().system('pip install nltk')


# In[28]:


nltk.download('punkt')


# In[31]:


df['num_characters'] = df['text'].apply(len)


# In[32]:


df.head()


# In[36]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[37]:


df.head()


# In[39]:


df['num_sentences'] = df['text'].apply(lambda x:nltk.sent_tokenize(x))


# In[40]:


df.head()


# In[41]:


df[['num_characters','num_words','num_sentences']].describe()


# In[44]:


df[df['label'] ==0][['num_characters','num_words','num_sentences']].describe()


# In[46]:


#spam
df[df['label'] == 1][['num_characters','num_words']].describe()


# In[47]:


import seaborn as sns


# In[54]:


sns.histplot(df[df['label'] == 0]['num_characters'])


# In[55]:


sns.pairplot(df,hue='label')


# ## 3. Data Preprocessing
#        -> Lower case
#        -> Tokenization
#        -> Removing special characters
#        -> Removing stop words and punctuation
#        -> Stemming

# In[81]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
        
    return " ".join(y)


# In[90]:


transform_text('Hi how Are you?')


# In[94]:


df['text'][10]


# In[96]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')


# In[98]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[99]:


df.head()


# In[134]:


get_ipython().system('pip install wordcloud')


# In[135]:


from wordcloud import WordCloud
wc = WordCloud(width=50, height=50, min_font_size=10, background_color='white')


# In[186]:


spam_wc = wc.generate(df[df['label'] == 1]['transformed_text'].str.cat(sep=""))


# In[188]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[ ]:





# In[ ]:




