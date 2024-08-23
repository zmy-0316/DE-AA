# Joint Extraction of Entities and Relations based on Word-Pair Distance Embedding and Axial Attention Mechanism

# 基于此对距离嵌入和轴向注意力机制的实体关系联合抽取模型


<img width="954" alt="image" src="https://github.com/user-attachments/assets/5d083359-8e5e-47bd-a8e3-65acf395a189">

The  joint extraction of entities and  relations provides key  technical  support  for  the construction of knowledge graphs, and the problem of overlapping relations has always been the focus of joint extraction model research. Many of  the existing methods use multi-step modeling methods. Although  they have achieved good results in solving the problem of overlapping relations, they have produced the problem of exposure bias. In order to solve the problem of overlapping relations and exposure bias at the same time, a joint entities and relations extraction method (DE-AA) based on word pair distance embedding and axial attention mechanism is proposed. Firstly, the table features of the representative word pair relation are constructed, and the word pair distance feature information is added to optimize its representation. Secondly, the axial attention model based on row attention and column attention is applied to enhance the table features, which can reduce the computational complexity while fusing  the  global  features.  Finally,  the  table  features  are mapped  to each  relation  space  to  generate  the relation-specific word pair relation table, and the table filling method is used to assign labels to each item in the table, and the triples are extracted by triple classification. The proposed model is evaluated on the public datasets NYT and WebNLG. The experimental results show  that  the proposed model achieves better performance  than other baseline models, and has significant advantages in dealing with overlapping relations or multiple relations. 


# running：run.py 
