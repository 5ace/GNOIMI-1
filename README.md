NovuGNO-IMI matching algorithm implementation
---------------------------------------------

This is the implementation of the paper ["Efficient Indexing of Billion-Scale datasets of deep descriptors"][1].  The implementation of `learn_GNOIMI.cpp` is based on the [original implementation][2] of the paper. Compared to the original implementation, current `learn_GNOIMI.cpp`  supports random thread number and random chunk size.

In current version, since we do not need to reconstruct the original feature vectors as described in the paper,  we did not use the re-ranking part of the paper (i.e., `search_GNOIMI.cpp`). Therefore, we had our own query function for the face matching. 


[1]: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf
[2]: https://github.com/arbabenko/GNOIMI
