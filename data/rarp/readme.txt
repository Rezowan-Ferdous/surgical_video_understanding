# the dataset was released in three parts:
# - Training set 1 (13 operations): videos 2, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17, 19
# - Training set 2 (27 operations): videos 1, 3, 7, 13, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 33, 34, 35, 36, 37, 38, 39, 40
# - Test set (10 operations): videos 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
# Videos 11, 15, 17 and 29, are split into two parts and annotated as video_xx_1 and video_xx_2.
# Such operations are treated as one during evaluation and we recommend to be used as one dataset
# when training action recognition models.
# | RGB value | Segmentation class                    |
# |-----------|---------------------------------------|
# | 1         | Other                                 |
# | 2         | Picking-up the needle                 |
# | 3         | Positioning the needle tip            |
# | 4         | Pushing the needle through the tissue |
# | 5         | Pulling the needle out of the tissue  |
# | 6         | Tying a knot                          |
# | 7         | Cutting the suture                    |
# | 8         | Returning/dropping the needle         |
# | RGB value | Segmentation class |
# |-----------|--------------------|
# | 0         | Background         |
# | 1         | Tool clasper       |
# | 2         | Tool wrist         |
# | 3         | Tool shaft         |
# | 4         | Suturing needle    |
# | 5         | Thread             |
# | 6         | Suction tool       |
# | 7         | Needle Holder      |
# | 8         | Clamps             |
# | 9         | Catheter           |
