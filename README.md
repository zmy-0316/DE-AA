# Joint Extraction of Entities and Relations based on Word-Pair Distance Embedding and Axial Attention Mechanism

*基于词对距离嵌入和轴向注意力机制的实体关系联合抽取模型*

### 模型整体框架图
<img width="954" alt="image" src="https://github.com/user-attachments/assets/5d083359-8e5e-47bd-a8e3-65acf395a189"><rb>

The  joint extraction of entities and  relations provides key  technical  support  for  the construction of knowledge graphs, and the problem of overlapping relations has always been the focus of joint extraction model research. Many of  the existing methods use multi-step modeling methods. Although  they have achieved good results in solving the problem of overlapping relations, they have produced the problem of exposure bias. In order to solve the problem of overlapping relations and exposure bias at the same time, a joint entities and relations extraction method (DE-AA) based on word pair distance embedding and axial attention mechanism is proposed. Firstly, the table features of the representative word pair relation are constructed, and the word pair distance feature information is added to optimize its representation. Secondly, the axial attention model based on row attention and column attention is applied to enhance the table features, which can reduce the computational complexity while fusing  the  global  features.  Finally,  the  table  features  are mapped  to each  relation  space  to  generate  the relation-specific word pair relation table, and the table filling method is used to assign labels to each item in the table, and the triples are extracted by triple classification. The proposed model is evaluated on the public datasets NYT and WebNLG. The experimental results show  that  the proposed model achieves better performance  than other baseline models, and has significant advantages in dealing with overlapping relations or multiple relations. <rb>

知识图谱的基本组成单位是包含实体和关系的三元组，当下常见的实体关系抽取模型有：实体关系pipeline建模、多任务学习、端到端联合抽取等，存在的问题：重叠关系抽取、误差传播、暴露偏差等，针对以上问题设计一种联合抽取模型，主要结构：bert预训练模型学习上下文特征，通过分析句法结构，加入词对距离嵌入特征，在构建特定标签集基础上，加入轴向注意力机制来更好的去学习三元组中头实体和尾实体间的对应关系标签，也更好的融合全局特征，降低计算复杂度。最终将特征映射到不同关系对应的特征空间中，采用表格填充法来抽取实体和关系。


### 表格填充法对应标签和标注示例
<img width="464" alt="image" src="https://github.com/user-attachments/assets/87dd07a2-3728-4005-9169-3354d36ebc7f"><rb>
* 标签集：label_list = ["N/A", "SB-MB", "SE-ME", "S-S", "MB-MB", "ME-ME", "MB-SB", "ME-SE"]<rb>
本文采用表格填充法去标注对应关系下的实体对（头实体和尾实体），结合序列标注的BIOES方法，标签集中的'-'用来分割头实体标签和尾实体标签，其中M和S指代是否是多词组成的实体，B和E指代当前位置是否是实体的边界词，即是开始边界还是结束边界。例如：'SE-ME'表示当前位置对应头实体是单个词构成的，且是头实体的开始位置，对应的尾实体则是多词组成的，且是尾实体的开始位置。这里注意：由于尾实体不是单个词组成的，所以虽然头实体是单个词构成的，但标记头实体的开始和结束位置的标签位于表格中的一行或一列，而不是同一个位置。具体细节可以看图。<rb>


* running：run.py 
