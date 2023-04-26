# CFNet
Include the muti-task joint learning network and the constraint fusion

Here you can learn a lot about CFNet, including the code of the multi-tasking joint learning network, code of the constraint fusion network, and more.
If you have any questions please contact us via Github.

For the code: We used Python 3.8 and tensorflow 2.0 to complete the programming of CFNet. The multi-task joint learning and constraint fusion are trained separately. Part 1 contains multi-task joint learning, which yields the corresponding three (global, local, and direct fusion) features. Part 2 contains constraint fusion, which requires the output from Part 1 as its input. The code for the experiments has been categorized according to the dataset. Notably, the code of the experiment is classified according to the dataset, which means that different datasets correspond to different code files.

For data set acquisition, please refer to the following literature:

CK+ dataset: Lucey, Patrick, et al. "The extended cohn-kanade dataset (ck+): A complete dataset for action unit and emotion-specified expression." 2010 ieee computer society conference on computer vision and pattern recognition-workshops. IEEE, 2010.

MMI dataset: Pantic, Maja, et al. "Web-based database for facial expression analysis." 2005 IEEE international conference on multimedia and Expo. IEEE, 2005.

RAF-DB dataset: Li, Shan, Weihong Deng, and JunPing Du. "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

When you need to learn or use this code for comparison or discussion, please cite the following literature:

Xiao, Junhao, et al. "CFNet: Facial expression recognition via constraint fusion under multi-task joint learning network." Applied Soft Computing (2023): 110312.
