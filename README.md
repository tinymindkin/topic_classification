## 功能

通过bert模型进行embedding，然后将embeding的到的vectors放到bertopic的fit_transfomer中进行训练，提取关键词，并根据词向量聚类分出多类别。

![](./assets/t-SNE Visualization.png)

![](./assets/3d_pca_visualization.png)

![](./assets/topkeywords.png)

主程序放在`bertt_topic.ipynb`文件内。

![](./assets/clustering_analysis.png)

## 参考

attention is all you need

embedding利用的是transfomer内部的embeding，可以学习到上下文特征。