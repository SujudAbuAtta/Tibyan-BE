from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt


dataset_df = pd.read_csv('Datasets/preprocessed_dataset.csv')

news = dataset_df['claim']
wordcloud = WordCloud(background_color="white",width=1600, height=800).generate(' '.join(news.tolist()))
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)

image = wordcloud.to_image()
image.show()