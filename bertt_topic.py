import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# Import necessary libraries
import torch
import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D

# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tokenizer
import json
import pickle
import pandas as pd
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer
import jieba
from plotly.subplots import make_subplots
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go



# Load the data
data = pd.read_csv('D:\proj\\fenglaifeng\______BERT__.csv')  

# Extract the text column
text_list = data['适合BERT聚类的格式总结']


with open('2停用词txt.txt', 'r', encoding='utf-8') as file:
    stopwords = set(file.read().splitlines())

def remove_stopwords(text, stopwords):
    return ' '.join([word for word in text.split() if word not in stopwords])

def tokenize_and_remove_stopwords(text, stopwords):
    tokens = jieba.lcut(text)
    return ' '.join([token for token in tokens if token not in stopwords])

# Apply tokenization and remove stopwords from the text_list
text_list = text_list.apply(lambda x: tokenize_and_remove_stopwords(x, stopwords))

# Initialize the BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)



# tokenize
inputs = tokenizer(text_list.to_list(), return_tensors="pt", padding=True, truncation=True)

#######多线性处理，谨慎使用
import concurrent.futures

def process_batch(start_idx, inputs, model):
    batch_size = 40
    batch = {k: v[start_idx:start_idx+batch_size] for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**batch)
    return output.last_hidden_state


outputs = []
batch_size = 40
num_threads = 8

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_batch, i, inputs, model) for i in range(0, len(inputs['input_ids']), batch_size)]
    for future in concurrent.futures.as_completed(futures):
        outputs.append(future.result())
        print(f"Processed batch {len(outputs)}")

outputs = torch.cat(outputs, dim=0)


### 下面是用来测试的
# with open('outputs_new_stopwords.pkl', 'wb') as file: 
#     pickle.dump(outputs, file)
# with open('outputs_new_stopwords.pkl', 'rb') as file:
#     outputs = pickle.load(file)

outputs = np.array(outputs)

last_hidden_state = outputs

#平均化
sentence_embeddings = outputs[:, 0, :]

# # Initialize the BERTopic model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize BERTopic with the embedding model
topic_model = BERTopic(embedding_model=embedding_model, nr_topics=20)


#####下面是预测
## bert ： bidirectional   tranformer ： 
topics, probabilities = topic_model.fit_transform(text_list.to_list(), sentence_embeddings)

# Extract the topics and their keywords
topic_info = topic_model.get_topic_info()
keywords = topic_model.get_topics()

# result_df
result_df = pd.DataFrame({
    '适合BERT聚类的格式总结': text_list,
    'Topic': topics,
    'Name': topic_info.set_index('Topic').loc[topics]['Name'].values,
})
result_df['主题词1'] = [keywords[topic][0][0] if topic in keywords and len(keywords[topic]) > 0 else '' for topic in topics]
result_df['主题词2'] = [keywords[topic][1][0] if topic in keywords and len(keywords[topic]) > 1 else '' for topic in topics]
result_df['主题词3'] = [keywords[topic][2][0] if topic in keywords and len(keywords[topic]) > 2 else '' for topic in topics]
result_df['主题词4'] = [keywords[topic][3][0] if topic in keywords and len(keywords[topic]) > 3 else '' for topic in topics]
result_df['主题词5'] = [keywords[topic][4][0] if topic in keywords and len(keywords[topic]) > 4 else '' for topic in topics]
result_df['主题词6'] = [keywords[topic][5][0] if topic in keywords and len(keywords[topic]) > 5 else '' for topic in topics]
result_df['主题词7'] = [keywords[topic][6][0] if topic in keywords and len(keywords[topic]) > 6 else '' for topic in topics]
result_df['主题词8'] = [keywords[topic][7][0] if topic in keywords and len(keywords[topic]) > 7 else '' for topic in topics]

#### ########ai绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
# sns.set(style="whitegrid")  # 美化样式

# 提取前 20 个关键词
top_n = 20
topic_keywords = topic_model.get_topic_info()
topic_keywords = topic_keywords[topic_keywords.Topic != -1]  # 过滤掉离群点

# 创建 DataFrame 存储关键词
keywords_df = pd.DataFrame()

for topic in topic_keywords.Topic:
    words = topic_model.get_topic(topic)
    words = [word[0] for word in words[:top_n]]
    keywords_df[f'主题 {topic}'] = words  # 主题编号

# 可视化前 5 个关键词
fig, axes = plt.subplots(len(topic_keywords), 1, figsize=(10, len(topic_keywords) * 8))

for i, topic in enumerate(topic_keywords.Topic):
    words = keywords_df[f'主题 {topic}'][:5].astype(str)  # 确保是字符串
    axes[i].barh(words, range(5, 0, -1), color='skyblue')  # 水平条形图
    axes[i].set_title(f'主题 {topic}', fontsize=12)  # 设置标题
    axes[i].invert_yaxis()  # 反转 y 轴，使排名高的排在上面

plt.tight_layout()
plt.savefig('D:\proj\\fenglaifeng\\topkeywords.png')


# ##############绘制主题相关性矩阵
fig = topic_model.visualize_heatmap()
fig.write_html('D:\proj\\fenglaifeng\\topic_correlation_matrix.html')


############ 绘制评论数据降维后聚类效果

# Reduce dimensions using PCA
pca = PCA(n_components=50)
pca_result = pca.fit_transform(sentence_embeddings)

# Further reduce dimensions using t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(pca_result)

# Create a DataFrame for visualization
tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
tsne_df['Topic'] = topics

# Plot the t-SNE result
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x='x', y='y',
    hue='Topic',
    palette=sns.color_palette('hsv', len(set(topics))),
    data=tsne_df,
    legend='full',
    alpha=0.3
)
plt.title('t-SNE visualization of BERT topic clusters')
plt.savefig('D:\proj\\fenglaifeng\\tsne_clusters.png')
plt.show()



# with open('D:\\proj\\fenglaifeng\\result_df.pkl', 'wb') as file:
#     pickle.dump(result_df, file)
# with open('D:\\proj\\fenglaifeng\\result_df.pkl', 'rb') as file:
#     result_df = pickle.load(file)






# Save the DataFrame to a CSV file
result_df.to_csv('BERT_topic_results_new.csv', index=False)





print(topic_info)
for topic, words in keywords.items():
    print(f"Topic {topic}: {words}")
