NovuGNO-IMI matching algorithm implementation
---------------------------------------------

This is the implementation of the paper ["Efficient Indexing of Billion-Scale datasets of deep descriptors"][1].  The implementation of `learn_GNOIMI.cpp` is based on the [original implementation][2] of the paper. Compared to the original implementation, current `learn_GNOIMI.cpp`  supports random thread number and random chunk size.

In current version, since we do not need to reconstruct the original feature vectors as described in the paper,  we did not use the re-ranking part of the paper (i.e., `search_GNOIMI.cpp`). Therefore, we had our own query function for the face matching. 

The implementation need [Yael library][3], which can be downloaded [here][4]. After compile the Yael library, we need to add the following line to the `~/.bashrc`:

    export YAEL_DIR= path/to/Yael




[1]: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf
[2]: https://github.com/arbabenko/GNOIMI
[3]: http://yael.gforge.inria.fr/
[4]: https://gforge.inria.fr/frs/download.php/file/34217/yael_v438.tar.gz
