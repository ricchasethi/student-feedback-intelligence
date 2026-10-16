# Day 4: BERTopic Theme Discovery and Taxonomy

## Chosen model

 n_topics  largest_topic_pct  outlier_pct  coherence_c_v  diversity  ari_vs_category
       16               13.2          9.0          0.496      0.806            0.336

## Topics

       count   pct  coherence_c_v                                                  description                                 theme
topic                                                                                                                               
0         87  12.0          0.600  Lecture delivery, interaction with students and punctuality                      Teaching Quality
1         77  10.6          0.551              Depth of courses and quality of course material                        Course Content
2         69   9.5          0.670          Exam pattern, marks distribution and paper checking                Assessment and Marking
3         67   9.3          0.431                 Library facilities and availability of books        Library and Learning Resources
4         63   8.7          0.390                                      Labs and practical work                        Practical Work
5         50   6.9          0.568                            Generic praise: good, best, great  General feedback (no specific theme)
6         36   5.0          0.526                        Extracurricular activities and events            Extracurricular Activities
7         36   5.0          0.568                    One-word praise: excellent, nice, perfect  General feedback (no specific theme)
8         36   5.0          0.519              Generic criticism: poor, bad, needs improvement  General feedback (no specific theme)
9         31   4.3          0.345                                                       'Good'  General feedback (no specific theme)
10        23   3.2          0.342                                  'Very good' / 'pretty good'  General feedback (no specific theme)
11        22   3.0          0.411                                         'Everything is fine'  General feedback (no specific theme)
12        19   2.6          0.585                                     'Satisfactory' / 'super'  General feedback (no specific theme)
13        15   2.1          0.359                         Praise for the university as a whole  General feedback (no specific theme)
14        15   2.1          0.535                                                    'Average'  General feedback (no specific theme)
15        13   1.8          0.538                              'They are doing their job well'  General feedback (no specific theme)

## Granularity

 nr_topics  n_topics  largest_topic_pct  outlier_pct  coherence_c_v  diversity  ari_vs_category
         8         7               42.2          6.1          0.543      0.893            0.130
        12        11               32.4          6.1          0.465      0.910            0.221
        15        14               23.8          6.1          0.463      0.806            0.286
        20        19               12.8          6.1          0.466      0.773            0.326

## Stability

 seed  n_topics  outlier_pct  ari_vs_seed_0
    0        16          7.0          1.000
    1        16          9.9          0.882
    2        16          9.9          0.855
    3        16         10.8          0.857
    4        16          9.8          0.876

## Themes vs existing categories

category                              course_content  examination  extracurricular  labwork  library_facilities  teaching
theme                                                                                                                    
Assessment and Marking                             1           62                1        4                   1         0
Course Content                                    72            1                1        0                   3         0
Extracurricular Activities                         0            0               35        0                   0         1
General feedback (no specific theme)              35           48               57       39                  43        38
Library and Learning Resources                     0            0                1        1                  65         0
Practical Work                                     0            1                0       62                   0         0
Teaching Quality                                   6            1                4        2                   0        74
Unassigned (outlier)                              10            9               19        9                   6        12

## Answers

1. ...
2. ...
3. ...
4. ...
5. ...
